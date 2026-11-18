import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models, transforms

DATA_DIR = "data/waterbirds_v1.0"
OUTPUT_DIR = "results"
SEED = 42

def load_models():
    # Load ResNet-50 backbone
    base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    base_model.eval()

    # Load saved linear heads
    heads_data = torch.load(os.path.join(OUTPUT_DIR, "linear_heads.pt"), weights_only=False)
    
    # ERM linear head
    erm_w = torch.tensor(heads_data["erm_coef"], dtype=torch.float32) # (1, 2048)
    erm_b = torch.tensor(heads_data["erm_intercept"], dtype=torch.float32) # (1,)
    
    # DFR linear head
    dfr_w = torch.tensor(heads_data["dfr_coef"], dtype=torch.float32) # (1, 2048)
    dfr_b = torch.tensor(heads_data["dfr_intercept"], dtype=torch.float32) # (1,)
    
    return base_model, (erm_w, erm_b), (dfr_w, dfr_b)

class ModelWithHead(nn.Module):
    def __init__(self, backbone, weight, bias):
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Linear(2048, 2)
        # Binary classification mapping to 2 classes:
        # logit_0 = -(w*x + b)/2, logit_1 = +(w*x + b)/2
        w_2class = torch.cat([-weight/2.0, weight/2.0], dim=0)
        b_2class = torch.cat([-bias/2.0, bias/2.0], dim=0)
        self.fc.weight = nn.Parameter(w_2class)
        self.fc.bias = nn.Parameter(b_2class)
        
    def forward(self, x):
        # We hook layer4
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        features = self.backbone.layer4(x)
        gap = self.backbone.avgpool(features)
        gap = torch.flatten(gap, 1)
        logits = self.fc(gap)
        return features, logits

def compute_gradcam(model, input_tensor, target_class):
    model.eval()
    features, logits = model(input_tensor)
    
    # Gradient of target class logit with respect to layer4 activations
    score = logits[0, target_class]
    grads = torch.autograd.grad(score, features, retain_graph=False)[0]
    
    # Global average pooling of gradients: alpha_k
    weights = torch.mean(grads, dim=(2, 3), keepdim=True) # (1, 2048, 1, 1)
    
    # Weighted combination of forward activation maps
    cam = torch.sum(weights * features, dim=1, keepdim=True) # (1, 1, H, W)
    cam = F.relu(cam)
    cam = cam.squeeze().detach().cpu().numpy()
    
    # Normalize to [0, 1]
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    else:
        cam = np.zeros_like(cam)
        
    return cam, logits.detach().cpu()

def main():
    base_model, erm_head, dfr_head = load_models()
    erm_full = ModelWithHead(base_model, erm_head[0], erm_head[1])
    dfr_full = ModelWithHead(base_model, dfr_head[0], dfr_head[1])

    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    unnormalize = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])
    ])

    df = pd.read_csv(os.path.join(DATA_DIR, "metadata.csv"))
    # Let's filter test set (split == 2)
    test_df = df[df['split'] == 2].reset_index(drop=True)
    test_df['group'] = 2 * test_df['y'] + test_df['place']

    # Focus on minority groups:
    # Group 2: Waterbird on Land (y=1, place=0) -> ERM often misclassifies as Landbird (0) because of Land background
    # Group 1: Landbird on Water (y=0, place=1) -> ERM often misclassifies as Waterbird (1) because of Water background
    g2_df = test_df[test_df['group'] == 2].reset_index(drop=True)
    g1_df = test_df[test_df['group'] == 1].reset_index(drop=True)

    print(f"Total Group 2 (Waterbird on Land) in test: {len(g2_df)}")
    print(f"Total Group 1 (Landbird on Water) in test: {len(g1_df)}")

    # Find compelling examples
    candidates_g2 = []
    for i in range(min(150, len(g2_df))):
        row = g2_df.iloc[i]
        img_path = os.path.join(DATA_DIR, row['img_filename'])
        img_pil = Image.open(img_path).convert('RGB')
        t_img = eval_transform(img_pil).unsqueeze(0)
        
        _, erm_logits = erm_full(t_img)
        _, dfr_logits = dfr_full(t_img)
        
        erm_pred = torch.argmax(erm_logits, dim=1).item()
        dfr_pred = torch.argmax(dfr_logits, dim=1).item()
        
        erm_prob_waterbird = F.softmax(erm_logits, dim=1)[0, 1].item()
        dfr_prob_waterbird = F.softmax(dfr_logits, dim=1)[0, 1].item()
        
        # We want: ERM predicted 0 (Landbird - wrong, fooled by land background)
        # and DFR predicted 1 (Waterbird - correct!)
        if erm_pred == 0 and dfr_pred == 1:
            candidates_g2.append({
                "idx": i,
                "path": row['img_filename'],
                "erm_prob": erm_prob_waterbird,
                "dfr_prob": dfr_prob_waterbird,
                "erm_pred": erm_pred,
                "dfr_pred": dfr_pred
            })

    print(f"Found {len(candidates_g2)} candidates in Group 2 where ERM failed and DFR succeeded!")

    # Let's also check Group 1 (Landbird on Water, y=0)
    candidates_g1 = []
    for i in range(min(150, len(g1_df))):
        row = g1_df.iloc[i]
        img_path = os.path.join(DATA_DIR, row['img_filename'])
        img_pil = Image.open(img_path).convert('RGB')
        t_img = eval_transform(img_pil).unsqueeze(0)
        
        _, erm_logits = erm_full(t_img)
        _, dfr_logits = dfr_full(t_img)
        
        erm_pred = torch.argmax(erm_logits, dim=1).item()
        dfr_pred = torch.argmax(dfr_logits, dim=1).item()
        
        erm_prob_landbird = F.softmax(erm_logits, dim=1)[0, 0].item()
        dfr_prob_landbird = F.softmax(dfr_logits, dim=1)[0, 0].item()
        
        # ERM predicted 1 (Waterbird - fooled by water background), DFR predicted 0 (Landbird - correct)
        if erm_pred == 1 and dfr_pred == 0:
            candidates_g1.append({
                "idx": i,
                "path": row['img_filename'],
                "erm_prob": erm_prob_landbird,
                "dfr_prob": dfr_prob_landbird,
                "erm_pred": erm_pred,
                "dfr_pred": dfr_pred
            })

    print(f"Found {len(candidates_g1)} candidates in Group 1 where ERM failed and DFR succeeded!")

    # Select the best example from Group 2 (Waterbird on Land)
    best_cand = candidates_g2[0] if candidates_g2 else candidates_g1[0]
    row = g2_df.iloc[best_cand['idx']] if candidates_g2 else g1_df.iloc[best_cand['idx']]
    img_path = os.path.join(DATA_DIR, row['img_filename'])
    img_pil = Image.open(img_path).convert('RGB')
    t_img = eval_transform(img_pil).unsqueeze(0)

    # Compute Grad-CAM for ERM (target = predicted class or ground truth class)
    # Ground truth: y=1 (Waterbird)
    target_class = row['y']
    erm_cam, erm_logits = compute_gradcam(erm_full, t_img, target_class=target_class)
    dfr_cam, dfr_logits = compute_gradcam(dfr_full, t_img, target_class=target_class)

    erm_probs = F.softmax(erm_logits, dim=1)[0].numpy()
    dfr_probs = F.softmax(dfr_logits, dim=1)[0].numpy()

    # Reconstruct display image
    disp_img = unnormalize(t_img[0]).permute(1, 2, 0).numpy()
    disp_img = np.clip(disp_img, 0, 1)

    # Resize CAM to 224x224
    from PIL import Image as PILImage
    erm_cam_resized = np.array(PILImage.fromarray(erm_cam).resize((224, 224), PILImage.Resampling.BILINEAR))
    dfr_cam_resized = np.array(PILImage.fromarray(dfr_cam).resize((224, 224), PILImage.Resampling.BILINEAR))

    # Generate Figure: 3 panels (Original, ERM Grad-CAM, DFR Grad-CAM)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Original Image
    axes[0].imshow(disp_img)
    class_label = "Waterbird" if row['y'] == 1 else "Landbird"
    bg_label = "Terra (Land)" if row['place'] == 0 else "Água (Water)"
    axes[0].set_title(f"Imagem Original\nClasse Real: {class_label} | Fundo: {bg_label}\n(Grupo Minoritário)", fontsize=11, fontweight='bold')
    axes[0].axis('off')

    # Panel 2: ERM Grad-CAM
    axes[1].imshow(disp_img)
    axes[1].imshow(erm_cam_resized, cmap='jet', alpha=0.55)
    erm_pred_label = "Waterbird" if np.argmax(erm_probs) == 1 else "Landbird"
    axes[1].set_title(f"Modelo ERM (Com Viés de Atalho)\nPredição: {erm_pred_label} (P(Water)={erm_probs[1]*100:.1f}%)\nFoco: Fundo / Elementos de Terra", fontsize=11, fontweight='bold', color='#b91c1c')
    axes[1].axis('off')

    # Panel 3: DFR Grad-CAM
    axes[2].imshow(disp_img)
    axes[2].imshow(dfr_cam_resized, cmap='jet', alpha=0.55)
    dfr_pred_label = "Waterbird" if np.argmax(dfr_probs) == 1 else "Landbird"
    axes[2].set_title(f"Modelo Mitigado DFR (Robusto)\nPredição: {dfr_pred_label} (P(Water)={dfr_probs[1]*100:.1f}%)\nFoco: Características da Ave", fontsize=11, fontweight='bold', color='#15803d')
    axes[2].axis('off')

    plt.suptitle("Interpretabilidade Visual: Grad-CAM no Grupo Minoritário (Waterbird em Fundo de Terra)", fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    gradcam_path = os.path.join(OUTPUT_DIR, "waterbirds_gradcam_comparison.png")
    plt.savefig(gradcam_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Grad-CAM salvo com sucesso em {gradcam_path}!")

    # Let's also create a multi-sample Grad-CAM comparison (2 minority examples: 1 Waterbird on Land, 1 Landbird on Water)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    examples = []
    if candidates_g2:
        examples.append((g2_df.iloc[candidates_g2[0]['idx']], "Waterbird em Terra (G2 - Minoria)"))
    if candidates_g1:
        examples.append((g1_df.iloc[candidates_g1[0]['idx']], "Landbird em Água (G1 - Minoria)"))

    for row_idx, (sample_row, group_desc) in enumerate(examples):
        sample_path = os.path.join(DATA_DIR, sample_row['img_filename'])
        sample_pil = Image.open(sample_path).convert('RGB')
        sample_t = eval_transform(sample_pil).unsqueeze(0)
        
        sample_target = sample_row['y']
        s_erm_cam, s_erm_logits = compute_gradcam(erm_full, sample_t, target_class=sample_target)
        s_dfr_cam, s_dfr_logits = compute_gradcam(dfr_full, sample_t, target_class=sample_target)
        
        s_erm_p = F.softmax(s_erm_logits, dim=1)[0].numpy()
        s_dfr_p = F.softmax(s_dfr_logits, dim=1)[0].numpy()
        
        s_disp = unnormalize(sample_t[0]).permute(1, 2, 0).numpy()
        s_disp = np.clip(s_disp, 0, 1)
        
        s_erm_cam_res = np.array(PILImage.fromarray(s_erm_cam).resize((224, 224), PILImage.Resampling.BILINEAR))
        s_dfr_cam_res = np.array(PILImage.fromarray(s_dfr_cam).resize((224, 224), PILImage.Resampling.BILINEAR))
        
        # Col 0: Original
        axes[row_idx, 0].imshow(s_disp)
        real_cls = "Waterbird" if sample_row['y'] == 1 else "Landbird"
        real_bg = "Terra" if sample_row['place'] == 0 else "Água"
        axes[row_idx, 0].set_title(f"Original: {group_desc}\nClasse: {real_cls} | Fundo: {real_bg}", fontsize=11, fontweight='bold')
        axes[row_idx, 0].axis('off')
        
        # Col 1: ERM
        axes[row_idx, 1].imshow(s_disp)
        axes[row_idx, 1].imshow(s_erm_cam_res, cmap='jet', alpha=0.55)
        pred_erm_str = "Waterbird" if np.argmax(s_erm_p) == 1 else "Landbird"
        status_erm = "INCORRETO" if np.argmax(s_erm_p) != sample_row['y'] else "CORRETO"
        axes[row_idx, 1].set_title(f"ERM: Pred={pred_erm_str} [{status_erm}]\n(Confiança Real={s_erm_p[sample_target]*100:.1f}%)", fontsize=11, fontweight='bold', color='#b91c1c')
        axes[row_idx, 1].axis('off')
        
        # Col 2: DFR
        axes[row_idx, 2].imshow(s_disp)
        axes[row_idx, 2].imshow(s_dfr_cam_res, cmap='jet', alpha=0.55)
        pred_dfr_str = "Waterbird" if np.argmax(s_dfr_p) == 1 else "Landbird"
        status_dfr = "CORRETO" if np.argmax(s_dfr_p) == sample_row['y'] else "INCORRETO"
        axes[row_idx, 2].set_title(f"DFR: Pred={pred_dfr_str} [{status_dfr}]\n(Confiança Real={s_dfr_p[sample_target]*100:.1f}%)", fontsize=11, fontweight='bold', color='#15803d')
        axes[row_idx, 2].axis('off')

    plt.suptitle("Comparação de Mapas de Atenção Grad-CAM: ERM (Atalho no Fundo) vs. DFR (Foco no Pássaro)", fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    multi_cam_path = os.path.join(OUTPUT_DIR, "waterbirds_gradcam_dual_minority.png")
    plt.savefig(multi_cam_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comparativo duplo Grad-CAM salvo em {multi_cam_path}!")

if __name__ == "__main__":
    main()
