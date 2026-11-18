import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

DATA_DIR = "data/waterbirds_v1.0"
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(DATA_DIR, "metadata.csv"))
df['group'] = 2 * df['y'] + df['place']

group_info = {
    0: {
        "name": "Grupo 0: Landbird em Terra",
        "type": "Maioria no Treino (72.95% - 3.498 amostras)",
        "color": "#1e3a8a",
        "status": "Correlação Espúria Alinhada"
    },
    1: {
        "name": "Grupo 1: Landbird em Água",
        "type": "Minoria no Treino (3.84% - 184 amostras)",
        "color": "#d97706",
        "status": "Correlação Violada (Desalinhado)"
    },
    2: {
        "name": "Grupo 2: Waterbird em Terra",
        "type": "Minoria Crítica no Treino (1.17% - 56 amostras!)",
        "color": "#dc2626",
        "status": "Pior Grupo no ERM (Gargalo)"
    },
    3: {
        "name": "Grupo 3: Waterbird em Água",
        "type": "Maioria no Treino (22.04% - 1.057 amostras)",
        "color": "#059669",
        "status": "Correlação Espúria Alinhada"
    }
}

# 1. Generate 4-panel image with representative examples
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
for g in range(4):
    g_samples = df[(df['split'] == 0) & (df['group'] == g)]
    # Pick a nice clear sample
    sample_row = g_samples.iloc[10 if len(g_samples) > 10 else 0]
    img_path = os.path.join(DATA_DIR, sample_row['img_filename'])
    img = Image.open(img_path).convert('RGB')
    
    axes[g].imshow(img)
    info = group_info[g]
    axes[g].set_title(f"{info['name']}\n{info['type']}\n[{info['status']}]", 
                      fontsize=11, fontweight='bold', pad=10)
    axes[g].axis('off')

plt.suptitle("Dataset Waterbirds (Sagawa et al., 2020) — Anatomia dos 4 Grupos (Classe × Fundo)", 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
overview_img_path = os.path.join(OUTPUT_DIR, "waterbirds_groups_examples.png")
plt.savefig(overview_img_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Exemplos visuais salvos em {overview_img_path}")

# 2. Generate Split distribution chart
fig, ax = plt.subplots(figsize=(10, 6))
split_map = {0: 'Treino (Train)', 1: 'Validação (Val)', 2: 'Teste (Test)'}
df['split_label'] = df['split'].map(split_map)
ct = pd.crosstab(df['split_label'], df['group'])
ct = ct.reindex(['Treino (Train)', 'Validação (Val)', 'Teste (Test)'])

# Calculate percentages within each split
ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

x = np.arange(len(ct.index))
width = 0.2
colors = ["#2563eb", "#f59e0b", "#ef4444", "#10b981"]
labels = [
    "G0: Landbird / Terra (Maioria)",
    "G1: Landbird / Água (Minoria)",
    "G2: Waterbird / Terra (Minoria)",
    "G3: Waterbird / Água (Maioria)"
]

for g in range(4):
    bars = ax.bar(x + (g - 1.5) * width, ct_pct[g], width, label=labels[g], color=colors[g])
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 1.0, f'{h:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylabel('Proporção Dentro do Split (%)', fontsize=12, fontweight='bold')
ax.set_title('Desbalanceamento Extremo no Treino vs. Equilíbrio no Teste (Waterbirds)', 
             fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(ct.index, fontsize=11, fontweight='bold')
ax.set_ylim(0, 85)
ax.legend(loc='upper right', fontsize=10)
ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
dist_chart_path = os.path.join(OUTPUT_DIR, "waterbirds_split_distribution.png")
plt.savefig(dist_chart_path, dpi=300)
plt.close()
print(f"Gráfico de distribuição salvo em {dist_chart_path}")
