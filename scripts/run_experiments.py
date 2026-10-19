import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Fixed seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Using device: {DEVICE}")

DATA_DIR = "data/waterbirds_v1.0"
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class WaterbirdsDataset(Dataset):
    def __init__(self, metadata_df, root_dir, transform=None):
        self.metadata_df = metadata_df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        row = self.metadata_df.iloc[idx]
        img_path = os.path.join(self.root_dir, row['img_filename'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        y = row['y']
        group = 2 * y + row['place']
        return image, y, group, idx

def extract_features():
    feature_file = os.path.join(OUTPUT_DIR, "waterbirds_resnet50_features.pt")
    if os.path.exists(feature_file):
        print(f"Loading existing features from {feature_file}...")
        return torch.load(feature_file, map_location="cpu", weights_only=False)

    print("Loading pretrained ResNet-50...")
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # Remove the final classification layer, keep avgpool
    modules = list(model.children())[:-1]
    feature_extractor = nn.Sequential(*modules)
    feature_extractor.to(DEVICE)
    feature_extractor.eval()

    # Standard ImageNet transform for evaluation
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    df = pd.read_csv(os.path.join(DATA_DIR, "metadata.csv"))
    dataset = WaterbirdsDataset(df, DATA_DIR, transform=eval_transform)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)

    features = []
    labels = []
    groups = []
    splits = []

    print(f"Extracting features for {len(dataset)} images on {DEVICE}...")
    start_t = time.time()
    with torch.no_grad():
        for i, (imgs, ys, gs, idxs) in enumerate(loader):
            imgs = imgs.to(DEVICE)
            feats = feature_extractor(imgs) # Shape: (B, 2048, 1, 1)
            feats = feats.flatten(1).cpu()
            features.append(feats)
            labels.append(ys)
            groups.append(gs)
            if (i + 1) % 20 == 0 or (i + 1) == len(loader):
                print(f"Processed batch {i+1}/{len(loader)} ({(i+1)*128}/{len(dataset)})...")

    features = torch.cat(features, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    groups = torch.cat(groups, dim=0).numpy()
    splits = df['split'].values

    data_dict = {
        "features": features,
        "labels": labels,
        "groups": groups,
        "splits": splits
    }
    torch.save(data_dict, feature_file)
    print(f"Features extracted in {time.time() - start_t:.1f}s and saved to {feature_file}!")
    return data_dict

def evaluate_predictions(y_true, y_pred, groups):
    """
    Computes Average Accuracy, Worst-Group Accuracy, and per-group accuracies.
    Groups:
      0: Landbird on Land (Majority)
      1: Landbird on Water (Minority)
      2: Waterbird on Land (Minority)
      3: Waterbird on Water (Majority)
    """
    avg_acc = accuracy_score(y_true, y_pred) * 100.0
    group_accs = {}
    for g in range(4):
        mask = (groups == g)
        if np.sum(mask) > 0:
            group_accs[g] = accuracy_score(y_true[mask], y_pred[mask]) * 100.0
        else:
            group_accs[g] = 0.0

    worst_group_acc = min(group_accs.values())
    disparity = max(group_accs.values()) - worst_group_acc

    return {
        "Avg_Acc": avg_acc,
        "Worst_Group_Acc": worst_group_acc,
        "Disparity": disparity,
        "Group_0_Land_Land": group_accs[0],
        "Group_1_Land_Water": group_accs[1],
        "Group_2_Water_Land": group_accs[2],
        "Group_3_Water_Water": group_accs[3]
    }

def main():
    data = extract_features()
    X = data["features"]
    y = data["labels"]
    g = data["groups"]
    s = data["splits"]

    # Splits: 0 = Train, 1 = Val, 2 = Test
    train_mask = (s == 0)
    val_mask = (s == 1)
    test_mask = (s == 2)

    X_train, y_train, g_train = X[train_mask], y[train_mask], g[train_mask]
    X_val, y_val, g_val = X[val_mask], y[val_mask], g[val_mask]
    X_test, y_test, g_test = X[test_mask], y[test_mask], g[test_mask]

    print(f"\nDados carregados:")
    print(f"Treino: {len(y_train)} amostras")
    print(f"Validação: {len(y_val)} amostras")
    print(f"Teste: {len(y_test)} amostras")

    results = {}
    models_dict = {}

    # -------------------------------------------------------------
    # 1. Baseline ERM (Empirical Risk Minimization)
    # Standard Logistic Regression trained on training set
    # -------------------------------------------------------------
    print("\n--- 1. Treinando Baseline ERM ---")
    erm_model = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
    erm_model.fit(X_train, y_train)
    y_test_pred_erm = erm_model.predict(X_test)
    results["ERM (Padrão)"] = evaluate_predictions(y_test, y_test_pred_erm, g_test)
    models_dict["ERM"] = erm_model

    # -------------------------------------------------------------
    # 2. Re-weighting Simples no Treino (Inverse Group Frequency)
    # Uses group annotations during training
    # -------------------------------------------------------------
    print("\n--- 2. Treinando Re-weighting de Grupos no Treino ---")
    # Compute inverse group weights
    group_counts = np.bincount(g_train, minlength=4)
    group_weights = 1.0 / group_counts
    sample_weights_train = group_weights[g_train]
    sample_weights_train /= sample_weights_train.mean() # normalize

    reweight_model = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
    reweight_model.fit(X_train, y_train, sample_weight=sample_weights_train)
    y_test_pred_reweight = reweight_model.predict(X_test)
    results["Re-weighting (Treino)"] = evaluate_predictions(y_test, y_test_pred_reweight, g_test)
    models_dict["Reweighting"] = reweight_model

    # -------------------------------------------------------------
    # 3. DFR (Deep Feature Reweighting - Kirichenko et al., 2022)
    # Retrains the last layer on a group-balanced subset of the validation set
    # -------------------------------------------------------------
    print("\n--- 3. Aplicando DFR (Deep Feature Reweighting na Validação) ---")
    # In validation set: group sizes are [467, 466, 133, 133].
    # Minority group size is 133.
    # We sample exactly 133 samples per group from validation set:
    val_indices = []
    min_group_size = min(np.bincount(g_val, minlength=4))
    print(f"DFR: Subamostrando {min_group_size} amostras por grupo do conjunto de validação...")
    rng = np.random.RandomState(SEED)
    for group_id in range(4):
        g_indices = np.where(g_val == group_id)[0]
        selected = rng.choice(g_indices, size=min_group_size, replace=False)
        val_indices.extend(selected)

    X_val_balanced = X_val[val_indices]
    y_val_balanced = y_val[val_indices]
    g_val_balanced = g_val[val_indices]

    # Hyperparameter tuning for C on validation WGA
    best_c = 1.0
    best_wga = -1
    for c_cand in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        clf = LogisticRegression(C=c_cand, max_iter=1000, random_state=SEED)
        clf.fit(X_val_balanced, y_val_balanced)
        # Evaluate on remaining val set
        val_preds = clf.predict(X_val)
        val_metrics = evaluate_predictions(y_val, val_preds, g_val)
        if val_metrics["Worst_Group_Acc"] > best_wga:
            best_wga = val_metrics["Worst_Group_Acc"]
            best_c = c_cand

    print(f"Melhor C para DFR: {best_c} (WGA na validação: {best_wga:.2f}%)")
    dfr_model = LogisticRegression(C=best_c, max_iter=1000, random_state=SEED)
    dfr_model.fit(X_val_balanced, y_val_balanced)
    y_test_pred_dfr = dfr_model.predict(X_test)
    results["DFR (Kirichenko et al.)"] = evaluate_predictions(y_test, y_test_pred_dfr, g_test)
    models_dict["DFR"] = dfr_model

    # -------------------------------------------------------------
    # 4. JTT (Just Train Twice - Liu et al., 2021)
    # Stage 1: Identification model (ERM). Identify misclassified training points.
    # Stage 2: Upweight misclassified points (error set) by lambda_up.
    # -------------------------------------------------------------
    print("\n--- 4. Aplicando JTT (Just Train Twice) ---")
    y_train_pred_erm = erm_model.predict(X_train)
    error_set_mask = (y_train_pred_erm != y_train)
    error_rate = error_set_mask.mean() * 100
    print(f"JTT: Taxa de erro do modelo de identificação no treino: {error_rate:.2f}% ({error_set_mask.sum()} amostras)")

    # Analyze composition of error set
    error_groups = g_train[error_set_mask]
    print(f"JTT: Composição do error set por grupo: {np.bincount(error_groups, minlength=4)}")

    # Stage 2: Upweight error set by lambda_up (e.g., 20)
    lambda_up = 20.0
    jtt_weights = np.ones(len(y_train))
    jtt_weights[error_set_mask] = lambda_up

    jtt_model = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
    jtt_model.fit(X_train, y_train, sample_weight=jtt_weights)
    y_test_pred_jtt = jtt_model.predict(X_test)
    results["JTT (Liu et al.)"] = evaluate_predictions(y_test, y_test_pred_jtt, g_test)
    models_dict["JTT"] = jtt_model

    # -------------------------------------------------------------
    # Exibir e Salvar Resultados
    # -------------------------------------------------------------
    res_df = pd.DataFrame(results).T
    cols = ["Avg_Acc", "Worst_Group_Acc", "Disparity", 
            "Group_0_Land_Land", "Group_1_Land_Water", 
            "Group_2_Water_Land", "Group_3_Water_Water"]
    res_df = res_df[cols]
    print("\n================== RESULTADOS REAIS NO CONJUNTO DE TESTE ==================")
    print(res_df.round(2).to_string())
    print("==========================================================================\n")

    csv_path = os.path.join(OUTPUT_DIR, "waterbirds_model_comparison.csv")
    res_df.to_csv(csv_path)
    print(f"Resultados salvos em {csv_path}")

    # Save models weights for Grad-CAM
    torch.save({
        "erm_coef": erm_model.coef_,
        "erm_intercept": erm_model.intercept_,
        "dfr_coef": dfr_model.coef_,
        "dfr_intercept": dfr_model.intercept_,
        "reweight_coef": reweight_model.coef_,
        "reweight_intercept": reweight_model.intercept_,
        "jtt_coef": jtt_model.coef_,
        "jtt_intercept": jtt_model.intercept_,
    }, os.path.join(OUTPUT_DIR, "linear_heads.pt"))

    # Plot comparison bar chart
    plt.figure(figsize=(10, 6))
    methods = res_df.index
    x = np.arange(len(methods))
    width = 0.35

    plt.bar(x - width/2, res_df["Avg_Acc"], width, label="Average Accuracy", color="#3b82f6")
    plt.bar(x + width/2, res_df["Worst_Group_Acc"], width, label="Worst-Group Accuracy (WGA)", color="#ef4444")

    for i in range(len(methods)):
        plt.text(x[i] - width/2, res_df["Avg_Acc"].iloc[i] + 1, f"{res_df['Avg_Acc'].iloc[i]:.1f}%", ha='center', fontweight='bold', fontsize=9)
        plt.text(x[i] + width/2, res_df["Worst_Group_Acc"].iloc[i] + 1, f"{res_df['Worst_Group_Acc'].iloc[i]:.1f}%", ha='center', fontweight='bold', fontsize=9, color="#b91c1c")

    plt.ylabel("Acurácia (%)", fontsize=12)
    plt.title("Comparação no Dataset Waterbirds: Acurácia Média vs. Pior Grupo (WGA)", fontsize=13, fontweight='bold', pad=15)
    plt.xticks(x, methods, fontsize=11)
    plt.ylim(0, 110)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(loc='lower right', fontsize=11)
    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, "waterbirds_accuracy_comparison.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Gráfico de barras salvo em {chart_path}")

    # Plot per-group breakdown chart
    plt.figure(figsize=(12, 6))
    group_labels = [
        "G0: Terra/Terra (Maioria)",
        "G1: Terra/Água (Minoria)",
        "G2: Água/Terra (Minoria)",
        "G3: Água/Água (Maioria)"
    ]
    group_cols = ["Group_0_Land_Land", "Group_1_Land_Water", "Group_2_Water_Land", "Group_3_Water_Water"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    bar_width = 0.18
    r = np.arange(len(group_labels))
    for idx, (method, color) in enumerate(zip(methods, colors)):
        vals = res_df.loc[method, group_cols].values
        plt.bar(r + idx * bar_width, vals, width=bar_width, label=method, color=color)

    plt.xlabel("Grupos de Teste (Classe / Fundo)", fontsize=12, fontweight='bold', labelpad=10)
    plt.ylabel("Acurácia (%)", fontsize=12, fontweight='bold')
    plt.title("Acurácia Detalhada por Grupo no Teste do Waterbirds", fontsize=13, fontweight='bold', pad=15)
    plt.xticks(r + bar_width * 1.5, group_labels, fontsize=10)
    plt.ylim(0, 115)
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    group_chart_path = os.path.join(OUTPUT_DIR, "waterbirds_group_accuracies.png")
    plt.savefig(group_chart_path, dpi=300)
    plt.close()
    print(f"Gráfico por grupos salvo em {group_chart_path}")

if __name__ == "__main__":
    main()
