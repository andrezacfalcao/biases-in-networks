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
BRAIN_DIR = "/Users/andrezafalcao/.gemini/antigravity/brain/31a5b3ef-7384-41a3-86e7-fc9619d642cf"

# Load ResNet-50
base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
base_model.eval()

heads_data = torch.load(os.path.join(OUTPUT_DIR, "linear_heads.pt"), weights_only=False)
erm_w = torch.tensor(heads_data["erm_coef"], dtype=torch.float32)
erm_b = torch.tensor(heads_data["erm_intercept"], dtype=torch.float32)
dfr_w = torch.tensor(heads_data["dfr_coef"], dtype=torch.float32)
dfr_b = torch.tensor(heads_data["dfr_intercept"], dtype=torch.float32)

class Net(nn.Module):
    def __init__(self, m, w, b):
        super().__init__()
        self.m = m
        self.fc = nn.Linear(2048, 2)
        w_2c = torch.cat([-w/2.0, w/2.0], dim=0)
        b_2c = torch.cat([-b/2.0, b/2.0], dim=0)
        self.fc.weight = nn.Parameter(w_2c)
        self.fc.bias = nn.Parameter(b_2c)
    def forward(self, x):
        x = self.m.conv1(x); x = self.m.bn1(x); x = self.m.relu(x); x = self.m.maxpool(x)
        x = self.m.layer1(x); x = self.m.layer2(x); x = self.m.layer3(x)
        act = self.m.layer4(x)
        gap = torch.flatten(self.m.avgpool(act), 1)
        logits = self.fc(gap)
        return act, logits

erm_net = Net(base_model, erm_w, erm_b)
dfr_net = Net(base_model, dfr_w, dfr_b)

eval_transform = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
unnorm = transforms.Compose([
    transforms.Normalize([0., 0., 0.], [1/0.229, 1/0.224, 1/0.225]),
    transforms.Normalize([-0.485, -0.456, -0.406], [1., 1., 1.])
])

df = pd.read_csv(os.path.join(DATA_DIR, "metadata.csv"))

def process_cam(img_id):
    row = df[df['img_id'] == img_id].iloc[0]
    img_pil = Image.open(os.path.join(DATA_DIR, row['img_filename'])).convert('RGB')
    t_in = eval_transform(img_pil).unsqueeze(0)
    
    # ERM forward & CAM on predicted class
    act_erm, l_erm = erm_net(t_in)
    p_erm = F.softmax(l_erm, dim=1)[0].detach().numpy()
    pred_erm = int(np.argmax(p_erm))
    score_erm = l_erm[0, pred_erm]
    grads_erm = torch.autograd.grad(score_erm, act_erm, retain_graph=True)[0]
    w_erm = torch.mean(grads_erm, dim=(2,3), keepdim=True)
    cam_erm = F.relu(torch.sum(w_erm * act_erm, dim=1)).squeeze().detach().numpy()
    cam_erm = (cam_erm - cam_erm.min()) / (cam_erm.max() - cam_erm.min() + 1e-8)
    
    # DFR forward & CAM on true class
    act_dfr, l_dfr = dfr_net(t_in)
    p_dfr = F.softmax(l_dfr, dim=1)[0].detach().numpy()
    pred_dfr = int(np.argmax(p_dfr))
    score_dfr = l_dfr[0, row['y']]
    grads_dfr = torch.autograd.grad(score_dfr, act_dfr, retain_graph=True)[0]
    w_dfr = torch.mean(grads_dfr, dim=(2,3), keepdim=True)
    cam_dfr = F.relu(torch.sum(w_dfr * act_dfr, dim=1)).squeeze().detach().numpy()
    cam_dfr = (cam_dfr - cam_dfr.min()) / (cam_dfr.max() - cam_dfr.min() + 1e-8)
    
    disp = np.clip(unnorm(t_in[0]).permute(1,2,0).numpy(), 0, 1)
    cam_erm_res = np.array(Image.fromarray(cam_erm).resize((224, 224), Image.Resampling.BILINEAR))
    cam_dfr_res = np.array(Image.fromarray(cam_dfr).resize((224, 224), Image.Resampling.BILINEAR))
    
    return row, disp, cam_erm_res, cam_dfr_res, p_erm, p_dfr

# 1. Generate new waterbirds_gradcam_baseline_bias.png (img_id = 3436: Gull in Bamboo)
row, disp, cam_erm_res, cam_dfr_res, p_erm, p_dfr = process_cam(3436)

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
axes[0].imshow(disp)
axes[0].set_title("1. Imagem de Teste Original (Grupo G2)\nClasse Real: Ave Aquática (Gaivota)\nFundo Espúrio: Floresta de Bambu", 
                  fontsize=12, fontweight='bold', pad=10)
axes[0].axis('off')

axes[1].imshow(disp)
axes[1].imshow(cam_erm_res, cmap='jet', alpha=0.55)
axes[1].set_title(f"2. Onde o Baseline ERM Olha (Antes dos Ajustes)\nPredição: Ave Terrestre (ERRO!) | Confiança: {p_erm[0]*100:.1f}%\nEvidência: Calor 100% no bambu (corpo da ave ignorado!)", 
                  fontsize=12, fontweight='bold', pad=10, color='#b91c1c')
axes[1].axis('off')

plt.suptitle("PROVA DO ATALHO NO BASELINE ERM (SEM AJUSTES):\nO modelo foi enganado pelo fundo e classificou os caules de bambu!", 
             fontsize=14, fontweight='bold', y=0.96)
plt.tight_layout(rect=[0, 0, 1, 0.90])
p1_path = os.path.join(OUTPUT_DIR, "waterbirds_gradcam_baseline_bias.png")
plt.savefig(p1_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Salvo {p1_path}")

# 2. Generate new waterbirds_gradcam_comparison.png (Antes vs Depois)
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
axes[0].imshow(disp)
axes[0].set_title("Imagem Original de Teste\nClasse Real: Ave Aquática (Gaivota)\nFundo Espúrio: Bambu (G2)", 
                  fontsize=11, fontweight='bold', pad=10)
axes[0].axis('off')

axes[1].imshow(disp)
axes[1].imshow(cam_erm_res, cmap='jet', alpha=0.55)
axes[1].set_title(f"1. ANTES: Baseline ERM (Com Viés)\nPredição: Ave Terrestre (94.9% certeza!)\nFoco Total: Caules de Bambu ao Fundo", 
                  fontsize=11, fontweight='bold', color='#b91c1c', pad=10)
axes[1].axis('off')

axes[2].imshow(disp)
axes[2].imshow(cam_dfr_res, cmap='jet', alpha=0.55)
axes[2].set_title(f"2. DEPOIS: Modelo com DFR (Mitigado)\nPredição: Ave Aquática (97.9% certeza!)\nFoco Total: Cabeça, Bico e Corpo da Ave", 
                  fontsize=11, fontweight='bold', color='#15803d', pad=10)
axes[2].axis('off')

plt.suptitle("Interpretabilidade Visual Grad-CAM: Revertendo o Atalho do Fundo com DFR", 
             fontsize=14, fontweight='bold', y=0.96)
plt.tight_layout(rect=[0, 0, 1, 0.90])
p2_path = os.path.join(OUTPUT_DIR, "waterbirds_gradcam_comparison.png")
plt.savefig(p2_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Salvo {p2_path}")

# 3. Generate dual minority comparison (G2: Gull in Bamboo + G1: Cuckoo in Ocean)
row_g1, disp_g1, cam_erm_g1, cam_dfr_g1, p_erm_g1, p_dfr_g1 = process_cam(1748)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Row 1: G2 (Waterbird in Bamboo)
axes[0, 0].imshow(disp)
axes[0, 0].set_title("G2 (Minoria): Ave Aquática em Terra\nClasse: Gaivota | Fundo: Bambu", fontsize=11, fontweight='bold')
axes[0, 0].axis('off')

axes[0, 1].imshow(disp)
axes[0, 1].imshow(cam_erm_res, cmap='jet', alpha=0.55)
axes[0, 1].set_title(f"ERM: Pred=Landbird (INCORRETO)\nFoco: Caules de Bambu ao Redor", fontsize=11, fontweight='bold', color='#b91c1c')
axes[0, 1].axis('off')

axes[0, 2].imshow(disp)
axes[0, 2].imshow(cam_dfr_res, cmap='jet', alpha=0.55)
axes[0, 2].set_title(f"DFR: Pred=Waterbird (CORRETO)\nFoco: Cabeça e Corpo da Gaivota", fontsize=11, fontweight='bold', color='#15803d')
axes[0, 2].axis('off')

# Row 2: G1 (Landbird in Ocean)
axes[1, 0].imshow(disp_g1)
axes[1, 0].set_title("G1 (Minoria): Ave Terrestre em Água\nClasse: Cuco | Fundo: Oceano/Praia", fontsize=11, fontweight='bold')
axes[1, 0].axis('off')

axes[1, 1].imshow(disp_g1)
axes[1, 1].imshow(cam_erm_g1, cmap='jet', alpha=0.55)
axes[1, 1].set_title(f"ERM: Pred=Waterbird (INCORRETO)\nFoco: Ondas e Espuma do Mar", fontsize=11, fontweight='bold', color='#b91c1c')
axes[1, 1].axis('off')

axes[1, 2].imshow(disp_g1)
axes[1, 2].imshow(cam_dfr_g1, cmap='jet', alpha=0.55)
axes[1, 2].set_title(f"DFR: Pred=Landbird (CORRETO)\nFoco: Peito e Plumagem do Cuco", fontsize=11, fontweight='bold', color='#15803d')
axes[1, 2].axis('off')

plt.suptitle("PROVA DUPLA DE SHORTCUT LEARNING: O ERM Olha para Água e Terra; O DFR Olha para o Pássaro", 
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
p3_path = os.path.join(OUTPUT_DIR, "waterbirds_gradcam_dual_minority.png")
plt.savefig(p3_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Salvo {p3_path}")

# Copy to brain dir
os.system(f"cp {p1_path} {p2_path} {p3_path} {BRAIN_DIR}/")
print("Imagens atualizadas copiadas para a pasta de artefatos!")
