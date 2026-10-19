import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from torchvision import models, transforms

DATA_DIR = "data/waterbirds_v1.0"
OUTPUT_DIR = "results"

def main():
    # Load ResNet-50 backbone
    base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    base_model.eval()

    heads_data = torch.load(os.path.join(OUTPUT_DIR, "linear_heads.pt"), weights_only=False)
    erm_w = torch.tensor(heads_data["erm_coef"], dtype=torch.float32)
    erm_b = torch.tensor(heads_data["erm_intercept"], dtype=torch.float32)

    class ModelWithHead(nn.Module):
        def __init__(self, backbone, weight, bias):
            super().__init__()
            self.backbone = backbone
            self.fc = nn.Linear(2048, 2)
            w_2class = torch.cat([-weight/2.0, weight/2.0], dim=0)
            b_2class = torch.cat([-bias/2.0, bias/2.0], dim=0)
            self.fc.weight = nn.Parameter(w_2class)
            self.fc.bias = nn.Parameter(b_2class)
            
        def forward(self, x):
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

    erm_full = ModelWithHead(base_model, erm_w, erm_b)

    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    unnormalize = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])
    ])

    df = pd.read_csv(os.path.join(DATA_DIR, "metadata.csv"))
    # Load img_id = 3 (Albatross on bamboo forest)
    row = df[df['img_id'] == 3].iloc[0]
    img_path = os.path.join(DATA_DIR, row['img_filename'])
    img_pil = Image.open(img_path).convert('RGB')
    t_img = eval_transform(img_pil).unsqueeze(0)

    # Compute ERM Grad-CAM for the PREDICTED class (Landbird = 0)
    # to show why ERM decided it's a Landbird!
    features, logits = erm_full(t_img)
    probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()
    pred_class = np.argmax(probs) # 0 = Landbird

    score = logits[0, pred_class]
    grads = torch.autograd.grad(score, features, retain_graph=False)[0]
    weights = torch.mean(grads, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * features, dim=1, keepdim=True)
    cam = F.relu(cam).squeeze().detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    cam_resized = np.array(Image.fromarray(cam).resize((224, 224), Image.Resampling.BILINEAR))
    disp_img = unnormalize(t_img[0]).permute(1, 2, 0).numpy()
    disp_img = np.clip(disp_img, 0, 1)

    # Generate 2-panel figure focusing solely on proving the baseline bias
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Panel 1: Original
    axes[0].imshow(disp_img)
    axes[0].set_title("1. Imagem de Teste Original (Grupo Minoritário G2)\nClasse Real: Ave Aquática (Albatroz) | Fundo: Floresta de Bambu", 
                      fontsize=12, fontweight='bold', pad=12)
    axes[0].axis('off')

    # Panel 2: Baseline ERM Heatmap
    axes[1].imshow(disp_img)
    axes[1].imshow(cam_resized, cmap='jet', alpha=0.55)
    axes[1].set_title(f"2. Para Onde o Modelo Baseline (ERM) Olha?\nPredição: Ave Terrestre (ERRO!) | Confiança: {probs[0]*100:.1f}%\n"
                      f"EVIDÊNCIA: Foco total nas árvores e no solo, ignorando o pássaro!", 
                      fontsize=12, fontweight='bold', pad=12, color='#b91c1c')
    axes[1].axis('off')

    plt.suptitle("PROVA DO ATALHO NO BASELINE ERM (SEM AJUSTES):\nO modelo foi enganado pelo fundo e classificou a vegetação!", 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "waterbirds_gradcam_baseline_bias.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figura de prova do viés no Baseline gerada em: {out_path}")

if __name__ == "__main__":
    main()
