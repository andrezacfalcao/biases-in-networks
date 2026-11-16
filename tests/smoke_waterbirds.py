"""Optional CPU integration check: one ERM epoch and DFR on 96 real images.

Run from the project root: python -m tests.smoke_waterbirds
Requires the reproduction dependencies, Waterbirds and the upstream checkout.
"""
import csv
import json
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

from reproduction.__main__ import commands, prepare
from reproduction.metrics import result_rows


def main():
    data = Path("data/waterbirds_v1.0").resolve()
    upstream = Path(".upstream/deep_feature_reweighting").resolve()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        work, sample, out = (root / name for name in ("work", "sample", "out"))
        for folder in (work, sample, out):
            folder.mkdir()
        prepare(upstream, work)
        taken = Counter()
        with (data / "metadata.csv").open() as source, (sample / "metadata.csv").open("w") as dest:
            reader = csv.DictReader(source)
            writer = csv.DictWriter(dest, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                key = (row["split"], row["y"], row["place"])
                if taken[key] == 8:
                    continue
                name = row["img_filename"]
                target = sample / name
                target.parent.mkdir(exist_ok=True, parents=True)
                target.symlink_to(data / name)
                writer.writerow(row)
                taken[key] += 1
        assert len(taken) == 12 and set(taken.values()) == {8}
        train, evaluate = commands(work, sample, out, "resnet50", 1)
        train.remove("--pretrained_model")
        train[train.index("--num_epochs") + 1] = "1"
        train[train.index("--batch_size") + 1] = "16"
        for stage, cmd in (("ERM", train), ("DFR", evaluate)):
            proc = subprocess.run([*cmd, "--device", "cpu"], cwd=work, capture_output=True, text=True)
            if proc.returncode:
                print(proc.stdout, proc.stderr)
                raise SystemExit(proc.returncode)
            print(f"{stage}: concluído", flush=True)
        raw = json.loads((out / "raw.pending.json").read_text())
        rows = result_rows(raw, dict.fromkeys(("train", "val", "test"), (8, 8, 8, 8)), "resnet50", 1)
        assert tuple(row["method"] for row in rows) == ("ERM", "DFR")
    print("Teste de integração em CPU. Nenhum resultado científico foi gravado.")


if __name__ == "__main__":
    main()
