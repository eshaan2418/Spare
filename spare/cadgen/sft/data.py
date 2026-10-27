"""Write SFT chat data (repr -> CadQuery program) for mlx_lm LoRA training.

Train: train-family records (data/synthetic/train.jsonl) + generic compositions (spare/synth/compose.py).
Valid: val-family records (hook), used only for early stopping / loss curves.
Test families never appear (DECISIONS D-006).
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from spare.cadgen.llm import SYSTEM
from spare.common.provenance import ROOT, read_jsonl


def to_chat(repr_text: str, program: str) -> dict:
    return {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": repr_text},
                         {"role": "assistant", "content": f"```python\n{program.strip()}\n```"}]}


SIMPLE = {"spacer", "cap_round", "knob_dshaft", "wheel_grooved", "compose_simple"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-family", type=int, default=24000)
    ap.add_argument("--n-compose", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/sft")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    fam = read_jsonl("data/synthetic/train.jsonl")
    assert all(r["split"] == "train" for r in fam)
    pick = rng.choice(len(fam), min(a.n_family, len(fam)), replace=False)
    rows = [(fam[int(i)]["family"], fam[int(i)]["repr"], fam[int(i)]["program"]) for i in pick]
    comp_path = ROOT / "data" / "synthetic" / "compose.jsonl"
    if comp_path.exists() and a.n_compose:
        comp = read_jsonl(comp_path)
        for i in rng.choice(len(comp), min(a.n_compose, len(comp)), replace=False):
            c = comp[int(i)]
            rows.append((c["family"], c["repr"], c["program"]))
    order = rng.permutation(len(rows))
    rows = [rows[i] for i in order]
    # curriculum: stage-1 file holds simple families only; stage-2 holds everything
    (out / "stage1").mkdir(exist_ok=True)
    with open(out / "stage1" / "train.jsonl", "w") as f:
        for fam_name, rp, pg in rows:
            if fam_name in SIMPLE:
                f.write(json.dumps(to_chat(rp, pg)) + "\n")
    (out / "stage2").mkdir(exist_ok=True)
    with open(out / "stage2" / "train.jsonl", "w") as f:
        for _, rp, pg in rows:
            f.write(json.dumps(to_chat(rp, pg)) + "\n")
    val = read_jsonl("data/benchmarks/val.jsonl")
    for st in ("stage1", "stage2"):
        with open(out / st / "valid.jsonl", "w") as f:
            for r in val[:60]:
                f.write(json.dumps(to_chat(r["repr"], r["program"])) + "\n")
    meta = {"n_rows": len(rows), "by_family": {k: int(v) for k, v in zip(*np.unique([r[0] for r in rows],
                                                                                      return_counts=True))}}
    (out / "META.json").write_text(json.dumps(meta, indent=1))
    print(meta)


if __name__ == "__main__":
    main()

