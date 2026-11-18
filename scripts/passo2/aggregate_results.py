"""Agrega runs/<arquitetura>/results.csv e calcula o efeito do DFR por semente.

Uso:
    python scripts/passo2/aggregate_results.py
    python scripts/passo2/aggregate_results.py --runs runs --output runs/aggregate

Gera summary_all.csv, deltas.csv e deltas_summary.csv. Arquiteturas sem os dois
métodos nas cinco sementes são ignoradas com aviso; o protocolo não permite
resumir sementes incompletas.
"""
import argparse
import csv
import sys
from collections import Counter, defaultdict
from math import sqrt
from pathlib import Path
from statistics import mean, stdev

SEEDS = (1, 2, 3, 4, 5)
METHODS = ("ERM", "DFR")
SCORES = ("worst_group_accuracy_pct", "weighted_accuracy_pct", "micro_accuracy_pct",
          "group_gap_pp", *(f"g{i}_accuracy_pct" for i in range(4)))
GROUPS = ("0_0", "0_1", "1_0", "1_1")
# t de Student bilateral, 4 graus de liberdade (cinco sementes).
T95_DF4 = 2.7764451051977987


def read_results(runs):
    rows = []
    for path in sorted(runs.glob("*/results.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"Sem results.csv em {runs}")
    return rows


def complete_architectures(rows):
    by_arch = defaultdict(dict)
    for row in rows:
        by_arch[row["architecture"]][row["method"], int(row["seed"])] = row
    complete = {}
    for architecture, runs in sorted(by_arch.items()):
        expected = {(method, seed) for method in METHODS for seed in SEEDS}
        missing = expected - set(runs)
        if missing:
            print(f"Aviso: {architecture} incompleta; faltam {len(missing)} execuções.",
                  file=sys.stderr)
            continue
        problems = [f"{method}/seed-{seed}" for (method, seed), row in runs.items()
                    if not all(row[score] for score in SCORES)]
        if problems:
            print(f"Aviso: {architecture} com métricas ausentes em {problems}; ignorada.",
                  file=sys.stderr)
            continue
        complete[architecture] = runs
    if not complete:
        raise SystemExit("Nenhuma arquitetura tem ERM e DFR nas cinco sementes.")
    return complete


def summarize(architecture, runs):
    summary, worst_groups = [], {}
    for method in METHODS:
        entries = [runs[method, seed] for seed in SEEDS]
        entry = {"architecture": architecture, "method": method, "n": len(entries)}
        for score in SCORES:
            values = [float(row[score]) for row in entries]
            entry[f"{score}_mean"] = mean(values)
            entry[f"{score}_sd"] = stdev(values)
        for score in ("worst_group_accuracy_pct", "weighted_accuracy_pct"):
            entry[f"{score}_ci95_pp"] = T95_DF4 * entry[f"{score}_sd"] / sqrt(len(entries))
        summary.append(entry)
        worst_groups[method] = Counter(
            GROUPS[min(range(4), key=lambda i: float(row[f"g{i}_accuracy_pct"]))]
            for row in entries)
    return summary, worst_groups


def deltas(architecture, runs):
    rows = []
    for seed in SEEDS:
        erm, dfr = runs["ERM", seed], runs["DFR", seed]
        erm_worst = float(erm["worst_group_accuracy_pct"])
        dfr_worst = float(dfr["worst_group_accuracy_pct"])
        rows.append({
            "architecture": architecture, "seed": seed,
            "erm_worst_group_pp": erm_worst, "dfr_worst_group_pp": dfr_worst,
            "delta_worst_group_pp": dfr_worst - erm_worst,
            "delta_weighted_pp": float(dfr["weighted_accuracy_pct"]) - float(erm["weighted_accuracy_pct"]),
            "delta_micro_pp": float(dfr["micro_accuracy_pct"]) - float(erm["micro_accuracy_pct"]),
            "erm_worst_group": GROUPS[min(range(4), key=lambda i: float(erm[f"g{i}_accuracy_pct"]))],
            "dfr_worst_group": GROUPS[min(range(4), key=lambda i: float(dfr[f"g{i}_accuracy_pct"]))],
        })
    return rows


def delta_summary(architecture, rows):
    values = [row["delta_worst_group_pp"] for row in rows]
    return {
        "architecture": architecture, "n": len(values),
        "delta_worst_group_mean_pp": mean(values),
        "delta_worst_group_sd_pp": stdev(values),
        "delta_worst_group_ci95_pp": T95_DF4 * stdev(values) / sqrt(len(values)),
        "delta_worst_group_min_pp": min(values),
        "delta_worst_group_max_pp": max(values),
        "delta_weighted_mean_pp": mean(row["delta_weighted_pp"] for row in rows),
        "delta_micro_mean_pp": mean(row["delta_micro_pp"] for row in rows),
    }


NUMERIC_PT = {
    "n_sementes", "pior_grupo_pct", "pior_grupo_desvio_pp", "pior_grupo_ic95_pp",
    "media_ponderada_pct", "media_ponderada_desvio_pp", "micro_pct",
    "micro_desvio_pp", "gap_pp", "delta_pior_grupo_pp",
    "delta_pior_grupo_desvio_pp", "delta_pior_grupo_ic95_pp",
    "delta_media_ponderada_pp", "delta_micro_pp",
}


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_excel_csv(path, rows):
    """CSV para Excel em português: separador ';', decimal com vírgula, BOM."""
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter=";")
        writer.writeheader()
        for row in rows:
            out = {}
            for key, value in row.items():
                if key in NUMERIC_PT and value != "":
                    out[key] = str(value).replace(".", ",")
                elif key == "dfr_c_valores" and value:
                    out[key] = value.replace(".", ",")
                else:
                    out[key] = value
            writer.writerow(out)


def wide_rows(architectures, summaries, delta_rows, complete):
    """Uma linha por arquitetura, com ERM e DFR lado a lado."""
    index = {(row["architecture"], row["method"]): row for row in summaries}
    wide = []
    for architecture in architectures:
        erm, dfr = index[architecture, "ERM"], index[architecture, "DFR"]
        delta = delta_rows[architecture]["summary"]
        runs = complete[architecture]

        def worst_group(method):
            counts = Counter(
                GROUPS[min(range(4), key=lambda i: float(runs[method, seed][f"g{i}_accuracy_pct"]))]
                for seed in SEEDS)
            return counts.most_common(1)[0][0]

        c_values = sorted({runs["DFR", seed]["dfr_c"] for seed in SEEDS}, key=float)
        row = {
            "arquitetura": architecture,
            "n_sementes": erm["n"],
            "erm_pior_grupo_pct": erm["worst_group_accuracy_pct_mean"],
            "erm_pior_grupo_desvio_pp": erm["worst_group_accuracy_pct_sd"],
            "erm_media_ponderada_pct": erm["weighted_accuracy_pct_mean"],
            "erm_media_ponderada_desvio_pp": erm["weighted_accuracy_pct_sd"],
            "erm_micro_pct": erm["micro_accuracy_pct_mean"],
            "erm_micro_desvio_pp": erm["micro_accuracy_pct_sd"],
            "erm_gap_pp": erm["group_gap_pp_mean"],
            "dfr_pior_grupo_pct": dfr["worst_group_accuracy_pct_mean"],
            "dfr_pior_grupo_desvio_pp": dfr["worst_group_accuracy_pct_sd"],
            "dfr_media_ponderada_pct": dfr["weighted_accuracy_pct_mean"],
            "dfr_media_ponderada_desvio_pp": dfr["weighted_accuracy_pct_sd"],
            "dfr_micro_pct": dfr["micro_accuracy_pct_mean"],
            "dfr_micro_desvio_pp": dfr["micro_accuracy_pct_sd"],
            "dfr_gap_pp": dfr["group_gap_pp_mean"],
            "delta_pior_grupo_pp": delta["delta_worst_group_mean_pp"],
            "delta_pior_grupo_desvio_pp": delta["delta_worst_group_sd_pp"],
            "delta_pior_grupo_ic95_pp": delta["delta_worst_group_ci95_pp"],
            "delta_media_ponderada_pp": delta["delta_weighted_mean_pp"],
            "delta_micro_pp": delta["delta_micro_mean_pp"],
            "erm_grupo_pior_predominante": worst_group("ERM"),
            "dfr_grupo_pior_predominante": worst_group("DFR"),
            "dfr_c_valores": " | ".join(c_values),
        }
        for i in range(4):
            row[f"acuracia_g{i}_erm"] = erm[f"g{i}_accuracy_pct_mean"]
            row[f"acuracia_g{i}_dfr"] = dfr[f"g{i}_accuracy_pct_mean"]
        wide.append(row)
    return wide


def summary_rows_pt(architectures, summaries, delta_rows, complete):
    """Tabela final em formato longo: uma linha por arquitetura e método."""
    index = {(row["architecture"], row["method"]): row for row in summaries}
    out = []
    for architecture in architectures:
        delta = delta_rows[architecture]["summary"]
        runs = complete[architecture]
        for method in METHODS:
            entry = index[architecture, method]
            groups = [float(entry[f"g{i}_accuracy_pct_mean"]) for i in range(4)]
            counts = Counter(
                GROUPS[min(range(4), key=lambda i: float(runs[method, seed][f"g{i}_accuracy_pct"]))]
                for seed in SEEDS)
            row = {
                "arquitetura": architecture,
                "metodo": method,
                "n_sementes": entry["n"],
                "pior_grupo_pct": entry["worst_group_accuracy_pct_mean"],
                "pior_grupo_desvio_pp": entry["worst_group_accuracy_pct_sd"],
                "pior_grupo_ic95_pp": entry["worst_group_accuracy_pct_ci95_pp"],
                "media_ponderada_pct": entry["weighted_accuracy_pct_mean"],
                "media_ponderada_desvio_pp": entry["weighted_accuracy_pct_sd"],
                "micro_pct": entry["micro_accuracy_pct_mean"],
                "micro_desvio_pp": entry["micro_accuracy_pct_sd"],
                "gap_pp": entry["group_gap_pp_mean"],
                "grupo_pior_predominante": counts.most_common(1)[0][0],
            }
            if method == "DFR":
                row.update({
                    "delta_pior_grupo_pp": delta["delta_worst_group_mean_pp"],
                    "delta_pior_grupo_desvio_pp": delta["delta_worst_group_sd_pp"],
                    "delta_pior_grupo_ic95_pp": delta["delta_worst_group_ci95_pp"],
                    "delta_media_ponderada_pp": delta["delta_weighted_mean_pp"],
                    "delta_micro_pp": delta["delta_micro_mean_pp"],
                    "dfr_c_valores": " | ".join(
                        sorted({runs["DFR", seed]["dfr_c"] for seed in SEEDS}, key=float)),
                })
            else:
                row.update({key: "" for key in (
                    "delta_pior_grupo_pp", "delta_pior_grupo_desvio_pp",
                    "delta_pior_grupo_ic95_pp", "delta_media_ponderada_pp",
                    "delta_micro_pp", "dfr_c_valores")})
            out.append(row)
    return out


def per_seed_rows(rows):
    """Detalhe por semente, com nomes de coluna em português."""
    out = []
    for row in sorted(rows, key=lambda r: (r["architecture"], int(r["seed"]), r["method"])):
        groups = [float(row[f"g{i}_accuracy_pct"]) for i in range(4)]
        out.append({
            "arquitetura": row["architecture"],
            "semente": row["seed"],
            "metodo": row["method"],
            "pior_grupo_pct": row["worst_group_accuracy_pct"],
            "grupo_pior": GROUPS[min(range(4), key=lambda i: groups[i])],
            "media_ponderada_pct": row["weighted_accuracy_pct"],
            "micro_pct": row["micro_accuracy_pct"],
            "gap_pp": row["group_gap_pp"],
            "acuracia_g0_pct": row["g0_accuracy_pct"],
            "acuracia_g1_pct": row["g1_accuracy_pct"],
            "acuracia_g2_pct": row["g2_accuracy_pct"],
            "acuracia_g3_pct": row["g3_accuracy_pct"],
            "dfr_c": row["dfr_c"],
        })
    return out


def markdown_table(architectures, summaries, delta_rows):
    lines = [
        "| Arquitetura | Método | Pior grupo (%) | Média ponderada (%) | Grupo pior |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    worst = {}
    for architecture in architectures:
        for entry in summaries:
            if entry["architecture"] != architecture:
                continue
            worst[architecture, entry["method"]] = entry
    for architecture in architectures:
        for method in METHODS:
            entry = worst[architecture, method]
            group = delta_rows[architecture]["worst_groups"][method].most_common(1)[0][0]
            lines.append(
                f"| {architecture} | {method} | {entry['worst_group_accuracy_pct_mean']:.1f} "
                f"± {entry['worst_group_accuracy_pct_sd']:.1f} | "
                f"{entry['weighted_accuracy_pct_mean']:.1f} ± "
                f"{entry['weighted_accuracy_pct_sd']:.1f} | {group} |")
    lines += ["", "| Arquitetura | Δ pior grupo (pp) | Δ média ponderada (pp) |",
              "| --- | ---: | ---: |"]
    for architecture in architectures:
        entry = delta_rows[architecture]["summary"]
        lines.append(f"| {architecture} | {entry['delta_worst_group_mean_pp']:+.1f} "
                     f"± {entry['delta_worst_group_sd_pp']:.1f} | "
                     f"{entry['delta_weighted_mean_pp']:+.1f} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, default=Path("runs/aggregate"))
    args = parser.parse_args()

    rows = read_results(args.runs)
    complete = complete_architectures(rows)
    architectures = list(complete)
    summaries, delta_rows = [], {}
    for architecture in architectures:
        summary, worst_groups = summarize(architecture, complete[architecture])
        summaries.extend(summary)
        per_seed = deltas(architecture, complete[architecture])
        delta_rows[architecture] = {
            "summary": delta_summary(architecture, per_seed),
            "worst_groups": worst_groups,
            "rows": per_seed,
        }

    write_csv(args.output / "summary_all.csv", summaries)
    write_csv(args.output / "results_all.csv",
              sorted(rows, key=lambda row: (row["architecture"], int(row["seed"]), row["method"])))
    write_csv(args.output / "deltas.csv",
              [row for architecture in architectures for row in delta_rows[architecture]["rows"]])
    write_csv(args.output / "deltas_summary.csv",
              [delta_rows[architecture]["summary"] for architecture in architectures])
    resumo_pt = summary_rows_pt(architectures, summaries, delta_rows, complete)
    write_csv(args.output / "resultados_resumo.csv", resumo_pt)
    write_excel_csv(args.output / "resultados_resumo_excel.csv", resumo_pt)
    write_csv(args.output / "resultados_completos.csv",
              wide_rows(architectures, summaries, delta_rows, complete))
    write_csv(args.output / "resultados_por_semente.csv", per_seed_rows(rows))
    table = markdown_table(architectures, summaries, delta_rows)
    (args.output / "resumo.txt").write_text(table, encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
