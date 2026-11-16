"""Waterbirds metadata and shared evaluation contract (percentages, not fractions)."""
import csv
import hashlib
import math
from collections import defaultdict
from statistics import mean, stdev

OFFICIAL_COUNTS = {
    "train": (3498, 184, 56, 1057),
    "val": (467, 466, 133, 133),
    "test": (2255, 2255, 642, 642),
}
GROUP_IDS = ("0_0", "0_1", "1_0", "1_1")
SCORES = ("worst_group_accuracy_pct", "weighted_accuracy_pct", "micro_accuracy_pct",
          "group_gap_pp", *(f"g{i}_accuracy_pct" for i in range(4)))
REFERENCE = {
    "ERM": {"worst_group_accuracy_pct": (74.9, 2.4), "weighted_accuracy_pct": (98.1, 0.1)},
    "DFR": {"worst_group_accuracy_pct": (92.9, 0.2), "weighted_accuracy_pct": (94.2, 0.4)},
}


def read_metadata(path):
    counts = {split: [0] * 4 for split in OFFICIAL_COUNTS}
    seen = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = row["img_filename"]
            if not name or name in seen:
                raise ValueError(f"Imagem vazia ou repetida nos metadados: {name!r}")
            seen.add(name)
            split, label, place = row["split"], row["y"], row["place"]
            if split not in ("0", "1", "2") or label not in ("0", "1") or place not in ("0", "1"):
                raise ValueError(f"Rótulo, fundo ou divisão inválida: {name}")
            counts[("train", "val", "test")[int(split)]][2 * int(label) + int(place)] += 1
    counts = {split: tuple(values) for split, values in counts.items()}
    if counts != OFFICIAL_COUNTS:
        raise ValueError(f"Contagens incompatíveis com as divisões oficiais: {counts}")
    return {"metadata_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "group_counts": counts}


def result_rows(raw, counts, architecture, seed):
    for split in ("train", "test"):
        if len(counts[split]) != 4 or min(counts[split]) <= 0:
            raise ValueError(f"{split} precisa conter os quatro grupos.")
    result = []
    for method, values in (
        ("ERM", [raw["erm"][f"accuracy_{group}"] for group in GROUP_IDS]),
        ("DFR", raw["dfr"]["test_accs"]),
    ):
        if len(values) != 4 or not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
            raise ValueError("Esperadas quatro acurácias entre 0 e 1.")
        groups = [v * 100 for v in values]
        result.append({
            "architecture": architecture, "method": method, "seed": seed,
            "worst_group_accuracy_pct": min(groups),
            "weighted_accuracy_pct": sum(a * n for a, n in zip(groups, counts["train"])) / sum(counts["train"]),
            "micro_accuracy_pct": sum(a * n for a, n in zip(groups, counts["test"])) / sum(counts["test"]),
            "group_gap_pp": max(groups) - min(groups),
            **{f"g{i}_accuracy_pct": value for i, value in enumerate(groups)},
            "dfr_c": raw["dfr"]["best_hypers"][0] if method == "DFR" else "",
        })
    return result


def summarize(rows, expected_seeds):
    pairs = defaultdict(dict)
    for row in rows:
        runs = pairs[row["architecture"], row["method"]]
        if row["seed"] in runs:
            raise ValueError("Combinação de arquitetura, método e semente duplicada.")
        runs[row["seed"]] = row
    if not pairs:
        raise ValueError("Sem resultados para resumir.")
    summary = []
    for (architecture, method), runs in sorted(pairs.items()):
        if set(runs) != set(expected_seeds):
            raise ValueError(f"Conjunto de sementes incompleto ou diferente: {architecture}/{method}")
        entry = {"architecture": architecture, "method": method, "n": len(runs)}
        for score in SCORES:
            values = [row[score] for row in runs.values()]
            entry[f"{score}_mean"] = mean(values)
            entry[f"{score}_sd"] = stdev(values) if len(values) > 1 else ""
        summary.append(entry)
    if any({method for arch, method in pairs if arch == a} != {"ERM", "DFR"} for a, _ in pairs):
        raise ValueError("Cada arquitetura precisa dos dois métodos: ERM e DFR.")
    return summary


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
