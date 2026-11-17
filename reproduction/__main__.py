"""Run the pinned Waterbirds protocol from the project root."""
import argparse
import csv
import hashlib
import json
import py_compile
import shlex
import shutil
import subprocess
import sys
import tempfile
from importlib import metadata
from pathlib import Path

from .metrics import REFERENCE, read_metadata, result_rows, summarize, write_csv

HERE = Path(__file__).resolve().parent
UPSTREAM = "9d8d5de78feafb0ba99878b8a1f91c5cd942e62a"
ARCHITECTURES = ("resnet50", "densenet121", "efficientnet_b0", "convnext_tiny", "vit_b_16",
                 "shufflenet_v2_x0_5", "mobilenet_v3_small", "mnasnet0_5")
SEEDS = (1, 2, 3, 4, 5)
UPSTREAM_FILES = ("LICENSE", "train_classifier.py", "dfr_evaluate_spurious.py", "wb_data.py", "utils.py")


def prepare(upstream, destination):
    for name in UPSTREAM_FILES:
        blob = subprocess.check_output(["git", "-C", str(upstream), "show", f"{UPSTREAM}:{name}"])
        (destination / name).write_bytes(blob)
    patch = HERE / "upstream.patch"
    subprocess.run(["git", "apply", "--whitespace=nowarn", str(patch)], cwd=destination, check=True)
    shutil.copyfile(HERE / "architectures.py", destination / "waterbirds_models.py")
    for script in destination.glob("*.py"):
        py_compile.compile(str(script), doraise=True)


def validate_data(data):
    path = data / "metadata.csv"
    info = read_metadata(path)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not (data / row["img_filename"]).is_file():
                raise ValueError(f"Imagem ausente: {row['img_filename']}")
    return info


def commands(work, data, out, architecture, seed):
    common = ["--architecture", architecture, "--data_dir", str(data), "--seed", str(seed)]
    return (
        [sys.executable, str(work / "train_classifier.py"), *common, "--test_wb_dir", str(data),
         "--output_dir", str(out / "erm"), "--pretrained_model", "--augment_data",
         "--num_epochs", "100", "--weight_decay", "0.001", "--batch_size", "32",
         "--momentum_decay", "0.9", "--init_lr", "0.001", "--eval_freq", "1"],
        [sys.executable, str(work / "dfr_evaluate_spurious.py"), *common,
         "--ckpt_path", str(out / "erm" / "final_checkpoint.pt"),
         "--result_path", str(out / "raw.pending.json")],
    )


def write_reports(out, dataset, architecture):
    rows, finished = [], []
    for seed in SEEDS:
        raw = out / f"seed-{seed}" / "raw.json"
        if raw.is_file():
            rows.extend(result_rows(json.loads(raw.read_text()), dataset["group_counts"], architecture, seed))
            finished.append(seed)
    write_csv(out / "results.csv", rows)
    if tuple(finished) != SEEDS:
        print(f"Sementes finalizadas: {finished}. O resumo exige {list(SEEDS)}.")
        return
    summary = summarize(rows, SEEDS)
    write_csv(out / "summary.csv", summary)
    if architecture == "resnet50":
        comparison = []
        for row in summary:
            for metric, (published_mean, published_sd) in REFERENCE[row["method"]].items():
                observed = row[f"{metric}_mean"]
                comparison.append({
                    "method": row["method"], "metric": metric, "n": row["n"],
                    "paper_mean_pct": published_mean, "paper_sd_pp": published_sd,
                    "observed_mean_pct": observed, "observed_sd_pp": row[f"{metric}_sd"],
                    "delta_pp": observed - published_mean,
                })
        write_csv(out / "paper_comparison.csv", comparison)


def run(work, data, output_root, dataset, architecture, seeds):
    import torch

    from .architectures import MODEL_SPECS

    if not torch.cuda.is_available():
        raise ValueError("Treino de referência requer CUDA. Use check neste computador e run em uma GPU NVIDIA.")
    out = output_root / architecture
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        **dataset,
        "profile": "kirichenko-iclr2023-waterbirds-table2",
        "architecture": architecture,
        "pretrained_weights": str(MODEL_SPECS[architecture][1]),
        "upstream_commit": UPSTREAM,
        "patch_sha256": hashlib.sha256((HERE / "upstream.patch").read_bytes()).hexdigest(),
        "model_adapter_sha256": hashlib.sha256((HERE / "architectures.py").read_bytes()).hexdigest(),
        "dependencies": {name: metadata.version(name) for name in (
            "torch", "torchvision", "numpy", "pandas", "scikit-learn", "pillow", "tensorboard")},
        "python": sys.version.split()[0],
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "seeds": SEEDS,
    }
    contents = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    manifest_path = out / "protocol.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != json.loads(contents):
        raise ValueError("A saída contém resultados com outro protocolo/ambiente. Use outro --output.")
    if not manifest_path.exists() and any(out.iterdir()):
        raise ValueError("A pasta já contém arquivos sem protocolo registrado. Use outro --output.")
    manifest_path.write_text(contents)
    for seed in seeds:
        seed_dir = out / f"seed-{seed}"
        raw = seed_dir / "raw.json"
        if raw.exists():
            print(f"Semente {seed} já concluída: {raw}", flush=True)
            continue
        seed_dir.mkdir(exist_ok=True)
        train, evaluate = commands(work, data, seed_dir, architecture, seed)
        with (seed_dir / "commands.txt").open("w") as handle:
            for command in (train, evaluate):
                handle.write(shlex.join(command) + "\n")
        # A finished ERM checkpoint is reusable if only DFR was interrupted.
        if not (seed_dir / "erm" / "final_checkpoint.pt").is_file():
            subprocess.run(train, cwd=work, check=True)
        pending = seed_dir / "raw.pending.json"
        pending.unlink(missing_ok=True)
        print(f"Avaliando ERM e DFR — semente {seed}. Log: {seed_dir / 'evaluation.log'}", flush=True)
        with (seed_dir / "evaluation.log").open("w") as log:
            subprocess.run(evaluate, cwd=work, stdout=log, stderr=subprocess.STDOUT, check=True)
        result_rows(json.loads(pending.read_text()), dataset["group_counts"], architecture, seed)
        pending.replace(raw)
        write_reports(out, dataset, architecture)
    write_reports(out, dataset, architecture)
    print(f"Resultados: {out / 'results.csv'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "run"))
    parser.add_argument("--data", type=Path, default=Path("data/waterbirds_v1.0"))
    parser.add_argument("--upstream", type=Path, default=Path(".upstream/deep_feature_reweighting"))
    parser.add_argument("--output", type=Path, default=Path("runs"))
    parser.add_argument("--architecture", choices=ARCHITECTURES, default="resnet50")
    parser.add_argument("--seed", type=int, choices=SEEDS, action="append",
                        help="Executar só esta semente; repetível. Padrão: 1 a 5.")
    args = parser.parse_args()
    try:
        dataset = validate_data(args.data.resolve())
        with tempfile.TemporaryDirectory(prefix="waterbirds-dfr-") as folder:
            work = Path(folder)
            prepare(args.upstream.resolve(), work)
            if args.action == "check":
                print(json.dumps({**dataset, "upstream_commit": UPSTREAM,
                                  "status": "dados, patch e sintaxe verificados; nenhum treino iniciado"}, indent=2))
                return
            run(work, args.data.resolve(), args.output.resolve(), dataset,
                args.architecture, sorted(set(args.seed or SEEDS)))
    except (OSError, ValueError, KeyError, ImportError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Erro: {error}\n")


if __name__ == "__main__":
    main()
