"""M4: program-generation baselines on frozen splits, scored by the held-out verifier suite (simulated).

Each system writes results/m4/systems/<name>.json (resumable); ``--summarize`` writes results/m4/baselines.json.

  python -m bench.m4_baselines --systems all --splits test,test_id --n 8
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from spare.cadgen.evaluate import (
    HELDOUT_FAST,
    HELDOUT_FULL,
    System,
    paired,
    run_system,
    summarize,
    summarize_by,
    template_generator,
)
from spare.cadgen.llm import BASE_1_5B, BASE_7B
from spare.common.provenance import ROOT, read_jsonl, stamp, write_result
from spare.verifiers.pool import VerifyPool


def systems() -> dict[str, System]:
    from spare.cadgen.generators import llm_generator as L

    S = {}
    # geometry-similarity references
    S["perfect_scan"] = System("perfect_scan", "similarity", template_generator("original"))
    S["copy_measured"] = System("copy_measured", "similarity", template_generator())
    # function-first design with an oracle program writer (upper bound for any generator on spare inputs)
    S["template_oracle"] = System("template_oracle", "spare", template_generator())
    for tag, mid in (("1.5b", BASE_1_5B), ("7b", BASE_7B)):
        S[f"llm{tag}_zs_sim"] = System(f"llm{tag}_zs_sim", "similarity", L(mid, input_mode="similarity"))
        S[f"llm{tag}_zs"] = System(f"llm{tag}_zs", "spare", L(mid, input_mode="spare"))
        S[f"llm{tag}_fs_bon"] = System(f"llm{tag}_fs_bon", "spare", L(mid, few_shot=True, input_mode="spare"),
                                       select="best_of_n")
        S[f"llm{tag}_fs"] = System(f"llm{tag}_fs", "spare", L(mid, few_shot=True, input_mode="spare"))
        S[f"llm{tag}_fs_sim"] = System(f"llm{tag}_fs_sim", "similarity", L(mid, few_shot=True,
                                                                          input_mode="similarity"))
    return S


OUT = ROOT / "results" / "m4"


def load_rows(name: str, base: Path = OUT / "systems") -> list[dict]:
    p = base / f"{name}.json"
    return json.loads(p.read_text())["rows"] if p.exists() else []


def summarize_all(base: Path = OUT / "systems", out: Path = OUT / "baselines.json", ref: str = "template_oracle",
                  seed: int = 0):
    res, all_rows = {}, {}
    for p in sorted(base.glob("*.json")):
        rows = json.loads(p.read_text())["rows"]
        all_rows[p.stem] = rows
        res[p.stem] = {"by_split": summarize_by(rows, "split", seed), "by_family": summarize_by(rows, "family", seed)}
    pairs = {}
    for name, rows in all_rows.items():
        if name == ref or ref not in all_rows:
            continue
        for split in ("test", "test_id"):
            a = [r for r in all_rows[ref] if r["split"] == split]
            b = [r for r in rows if r["split"] == split]
            for m in ("fit1", "fit2", "all_pass1"):
                d = paired(a, b, m, seed)
                if d:
                    pairs[f"{ref} - {name} [{split}] {m}"] = d
    write_result(out, {"systems": res, "paired_vs_" + ref: pairs}, stamp(seed, {"base": str(base)}))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="all")
    ap.add_argument("--splits", default="test,test_id")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="records per split (0 = all)")
    ap.add_argument("--fast", action="store_true", help="skip slicer/FEA/functional-edit (smoke tests only)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--out", default=str(OUT / "systems"))
    a = ap.parse_args()
    base = Path(a.out)
    if a.summarize:
        summarize_all(base)
        return
    recs = []
    for s in a.splits.split(","):
        rr = read_jsonl(f"data/benchmarks/{s}.jsonl")
        recs += rr[: a.limit] if a.limit else rr
    S = systems()
    names = list(S) if a.systems == "all" else a.systems.split(",")
    base.mkdir(parents=True, exist_ok=True)
    with VerifyPool(timeout=180) as pool:
        for nm in names:
            p = base / f"{nm}.json"
            if p.exists() and not a.force:
                print("skip (exists)", nm)
                continue
            t0 = time.time()
            rows = run_system(S[nm], recs, pool, n=a.n, seed=a.seed,
                              heldout_opts=HELDOUT_FAST if a.fast else HELDOUT_FULL)
            sysd = S[nm]
            write_result(p, {"system": nm, "input": sysd.input, "select": sysd.select, "n": a.n,
                             "seconds": round(time.time() - t0, 1), "summary": summarize(rows),
                             "by_split": summarize_by(rows, "split"), "rows": rows},
                         stamp(a.seed, {**vars(a), "system": nm}))
    summarize_all(base)


if __name__ == "__main__":
    main()
