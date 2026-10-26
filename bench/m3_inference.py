"""M3: measurement fusion + mating inference + tolerance engine, evaluated with the oracle program generator.

The *program* is the family template rendered on the inferred design dims, so this isolates design
inference (contribution 1) from code generation. Fit is judged by the held-out fit verifier against the
TRUE mating geometry under (a) the default printer the tolerance engine assumes ("known printer") and
(b) the part's assigned hidden virtual printer ("unknown printer").
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from spare.common.provenance import bootstrap_ci, paired_bootstrap_diff, read_jsonl, stamp, write_result
from spare.features.infer import CONDITIONS, infer_record, links
from spare.synth.registry import get as get_family
from spare.verifiers.pool import VerifyJob, VerifyPool
from spare.verifiers.suite import virtual_printer_for

CONFIGS = [(c, "full", True) for c in CONDITIONS] + [
    ("caliper", "no_tolerance", True), ("caliper", "no_mating", True),
    ("no_mating_item", "no_tolerance", True), ("photo_only", "full", False), ("no_mating_item", "full", False)]
FAST = {"slicer": False, "fea": False, "functional_edit": False}


def cfg_name(c):
    return f"{c[0]}/{c[1]}" + ("" if c[2] else "/no_std_sizes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="test,val,test_id")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="results/m3/inference.json")
    a = ap.parse_args()
    recs = []
    for s in a.splits.split(","):
        recs += read_jsonl(f"data/benchmarks/{s}.jsonl")
    if a.limit:
        recs = recs[: a.limit]
    jobs, meta = [], []
    for r in recs:
        fam = get_family(r["family"])
        for c in CONFIGS:
            x = infer_record(r, c[0], c[1], a.seed, standard_sizes=c[2])
            src = fam.program(x["design_hat"], r["style"])
            for pname in ("default", "virtual"):
                pr = None if pname == "default" else virtual_printer_for(r["uid"]).name
                jobs.append(VerifyJob(src, r, "heldout", pr, opts=FAST))
                meta.append((r, c, pname, x))
    t0 = time.time()
    with VerifyPool(timeout=120) as pool:
        reps = pool.run(jobs)
    rows = []
    for (r, c, pname, x), rep in zip(meta, reps):
        lk = links(r["family"])
        merr = {q: abs(x["obs_mating"][q] - r["mating"][q]) for q in r["mating"] if q != "load_N"}
        derr = {p: abs(x["design_hat"][p] - r["design"][p]) for p in lk}
        rows.append({"uid": r["uid"], "family": r["family"], "split": r["split"], "cfg": cfg_name(c),
                     "printer": pname, "fit_pass": bool(rep.get("fit_pass", False)),
                     "all_pass": bool(rep.get("all_pass", False)), "valid": bool(rep.get("valid", False)),
                     "mating_abs_err": merr, "critical_design_abs_err": derr,
                     "mating_src": {q: x["obs_source"].get("m:" + q, "") for q in merr},
                     "fit_verdicts": [f.get("verdict") for f in rep.get("fit_detail", []) if "verdict" in f]})
    summary: dict = {}
    for split_group in ("test", "val", "test_id"):
        g = {}
        for c in CONFIGS:
            cn = cfg_name(c)
            for pname in ("default", "virtual"):
                rr = [x for x in rows if x["cfg"] == cn and x["printer"] == pname and x["split"] == split_group]
                if not rr:
                    continue
                pt, lo, hi = bootstrap_ci([x["fit_pass"] for x in rr], seed=a.seed)
                g[f"{cn}@{pname}"] = {"n": len(rr), "fit_pass": pt, "ci95": [lo, hi]}
            rr = [x for x in rows if x["cfg"] == cn and x["printer"] == "default" and x["split"] == split_group]
            me = [v for x in rr for v in x["mating_abs_err"].values()]
            de = [v for x in rr for v in x["critical_design_abs_err"].values()]
            if rr:
                g[f"{cn}:errors"] = {"mating_abs_err_median": float(np.median(me)) if me else None,
                                     "mating_abs_err_p90": float(np.quantile(me, 0.9)) if me else None,
                                     "critical_design_abs_err_median": float(np.median(de)) if de else None,
                                     "critical_design_abs_err_p90": float(np.quantile(de, 0.9)) if de else None}
        summary[split_group] = g
    # paired comparisons on unseen test families: full vs ablations (same parts)
    paired = {}
    for pname in ("default", "virtual"):
        base = {x["uid"]: x["fit_pass"] for x in rows if x["cfg"] == "caliper/full" and x["printer"] == pname
                and x["split"] == "test"}
        for c in CONFIGS:
            cn = cfg_name(c)
            if cn == "caliper/full":
                continue
            oth = {x["uid"]: x["fit_pass"] for x in rows if x["cfg"] == cn and x["printer"] == pname
                   and x["split"] == "test"}
            u = sorted(set(base) & set(oth))
            if u:
                d, lo, hi, p = paired_bootstrap_diff([base[k] for k in u], [oth[k] for k in u], seed=a.seed)
                paired[f"caliper/full - {cn} @{pname}"] = {"n": len(u), "diff": d, "ci95": [lo, hi], "p": p}
    # mating error by source
    by_src: dict[str, list[float]] = {}
    for x in rows:
        if x["printer"] != "default":
            continue
        for q, e in x["mating_abs_err"].items():
            by_src.setdefault(x["mating_src"][q] or "?", []).append(e)
    src_summary = {k: {"n": len(v), "median": float(np.median(v)), "p90": float(np.quantile(v, 0.9))}
                   for k, v in by_src.items()}
    write_result(a.out, {"configs": [cfg_name(c) for c in CONFIGS], "n_records": len(recs),
                         "seconds": round(time.time() - t0, 1), "summary": summary, "paired_test": paired,
                         "mating_err_by_source": src_summary, "rows": rows}, stamp(a.seed, vars(a)))
    for k, v in summary["test"].items():
        print("test", k, v)
    for k, v in paired.items():
        print(k, v)


if __name__ == "__main__":
    main()
