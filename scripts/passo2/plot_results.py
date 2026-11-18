"""Gera a figura de comparação ERM × DFR a partir de runs/aggregate/summary_all.csv (ou results/passo2/summary_all.csv)."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = {
    "shufflenet_v2_x0_5": "ShuffleNet V2 ×0,5",
    "mobilenet_v3_small": "MobileNetV3-Small",
    "efficientnet_b0": "EfficientNet-B0",
    "densenet121": "DenseNet-121",
    "resnet50": "ResNet-50",
}
ORDER = list(LABELS)
COLORS = {"ERM": "#4C72B0", "DFR": "#DD8452"}


def load(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    data = {}
    for row in rows:
        data[row["architecture"], row["method"]] = row
    architectures = [a for a in ORDER if (a, "ERM") in data and (a, "DFR") in data]
    if not architectures:
        raise SystemExit(f"Sem pares ERM/DFR em {path}")
    return data, architectures


def panel(ax, data, architectures, score, title, ylabel, ylim):
    x = np.arange(len(architectures))
    width = 0.36
    for offset, method in ((-width / 2, "ERM"), (width / 2, "DFR")):
        means = [float(data[a, method][f"{score}_mean"]) for a in architectures]
        errors = [float(data[a, method][f"{score}_sd"]) for a in architectures]
        ax.bar(x + offset, means, width, yerr=errors, capsize=3,
               label=method, color=COLORS[method], edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[a] for a in architectures], rotation=18, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=Path("runs/aggregate/summary_all.csv"))
    parser.add_argument("--output", type=Path, default=Path("runs/aggregate/comparacao.png"))
    args = parser.parse_args()

    data, architectures = load(args.summary)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    panel(axes[0], data, architectures, "worst_group_accuracy_pct",
          "Acurácia do pior grupo", "Acurácia (%)", (0, 100))
    panel(axes[1], data, architectures, "weighted_accuracy_pct",
          "Acurácia média ponderada", "Acurácia (%)", (60, 100))
    fig.suptitle("Waterbirds — ERM e DFR por arquitetura (5 sementes, média ± desvio)")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(args.output)


if __name__ == "__main__":
    main()
