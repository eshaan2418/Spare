"""Build the synthetic dataset and frozen benchmark splits.

Outputs
  data/synthetic/train.jsonl      >= N programs from train families (git-ignored; regenerable from seed)
  data/benchmarks/{val,test,test_id}.jsonl   frozen evaluation sets (committed)
  data/benchmarks/MANIFEST.json   sha256 checksums, counts, seed, git commit
  results/m2/dataset_stats.json   execution/validity stats
Every record: uid, family, split, style, material, mating, original, design, repr (text), program.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from spare.common.sandbox import ProgramRunner
from spare.repr.render import render
from spare.synth.registry import FAMILIES, families_in

ROOT = Path(__file__).resolve().parents[2]


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_record(fam, rng, uid: str, split: str) -> dict:
    material = "PETG" if rng.random() < 0.3 else "PLA"
    inst = fam.instance(rng, uid, material=material)
    inst.split = split
    graph = fam.feature_graph(inst.mating, inst.design, inst.original)
    prof = "generic_petg_0.4" if material == "PETG" else "generic_pla_0.4"
    return {
        "uid": uid, "family": fam.name, "split": split, "style": inst.style, "material": material,
        "mating": inst.mating, "original": inst.original, "design": inst.design,
        "repr": render(graph, material, prof), "program": fam.program(inst.design, inst.style),
    }


def build(n_train: int, seed: int, out_root: Path, verify_frac: float = 1.0, workers: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    syn = out_root / "data" / "synthetic"
    bench = out_root / "data" / "benchmarks"
    syn.mkdir(parents=True, exist_ok=True)
    bench.mkdir(parents=True, exist_ok=True)
    splits: dict[str, list[dict]] = {"train": [], "test_id": [], "val": [], "test": []}
    train_f = families_in("train")
    per = int(np.ceil(n_train / len(train_f)))
    for f in train_f:
        for i in range(per):
            splits["train"].append(make_record(f, rng, f"{f.name}_tr{i:05d}", "train"))
        for i in range(25):  # in-distribution test (held-out instances of train families)
            splits["test_id"].append(make_record(f, rng, f"{f.name}_tid{i:03d}", "test_id"))
    for f in families_in("val"):
        for i in range(100):
            splits["val"].append(make_record(f, rng, f"{f.name}_va{i:03d}", "val"))
    for f in families_in("test"):
        for i in range(60):
            splits["test"].append(make_record(f, rng, f"{f.name}_te{i:03d}", "test"))
    # execution check (drops anything that fails; reports counts)
    stats: dict = {"seed": seed, "families": {k: v.split for k, v in FAMILIES.items()}}
    t0 = time.time()
    with ProgramRunner(workers=workers, timeout=30) as runner:
        for name, recs in splits.items():
            idx = list(range(len(recs)))
            if name == "train" and verify_frac < 1.0:
                idx = sorted(rng.choice(len(recs), int(len(recs) * verify_frac), replace=False).tolist())
            ok = np.ones(len(recs), bool)
            B = 512
            for b in range(0, len(idx), B):
                chunk = idx[b:b + B]
                res = runner.run_many([recs[i]["program"] for i in chunk], trusted=True)
                for i, r in zip(chunk, res):
                    if not r.ok:
                        ok[i] = False
            stats[name] = {"generated": len(recs), "checked": len(idx), "exec_fail": int((~ok).sum())}
            splits[name] = [r for r, k in zip(recs, ok) if k]
            stats[name]["kept"] = len(splits[name])
    stats["exec_seconds"] = round(time.time() - t0, 1)
    paths = {"train": syn / "train.jsonl", "test_id": bench / "test_id.jsonl", "val": bench / "val.jsonl",
             "test": bench / "test.jsonl"}
    for name, p in paths.items():
        with open(p, "w") as fh:
            for r in splits[name]:
                fh.write(json.dumps(r) + "\n")
    manifest = {"seed": seed, "git_commit": git_commit(), "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "files": {name: {"path": str(p.relative_to(out_root)), "sha256": sha256(p), "n": len(splits[name])}
                          for name, p in paths.items()},
                "split_by_family": {s: sorted(f.name for f in families_in(s)) for s in ("train", "val", "test")}}
    (bench / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    res_dir = out_root / "results" / "m2"
    res_dir.mkdir(parents=True, exist_ok=True)
    stats["manifest"] = manifest
    (res_dir / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verify-frac", type=float, default=1.0)
    ap.add_argument("--out", type=str, default=str(ROOT))
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    s = build(a.n, a.seed, Path(a.out), a.verify_frac, a.workers)
    print(json.dumps({k: v for k, v in s.items() if k != "manifest"}, indent=1))


if __name__ == "__main__":
    main()
