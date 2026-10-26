"""M2: agreement between primary (reward) and held-out (evaluation-only) verifiers.

For a sample of frozen benchmark parts we evaluate the oracle program plus family-agnostic mutations
(critical parameters pushed tight/loose by random amounts, borderline nudges, random rescaling of other
parameters). Both suites use the *same* printer here so disagreement measures verifier error, not printer
mismatch. Reports per-component agreement and Cohen's kappa with bootstrap CIs over parts.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from spare.common.provenance import bootstrap_ci, read_jsonl, stamp, write_result
from spare.synth.registry import get as get_family
from spare.verifiers.pool import VerifyJob, VerifyPool

COMPONENTS = ["valid", "fit_pass", "dims_pass", "print_pass", "strength_pass", "all_pass"]
THRESHOLD = 0.80  # pre-registered minimum agreement per component (DECISIONS D-016)


def mutate(rec: dict, rng: np.random.Generator, kind: str) -> dict:
    fam = get_family(rec["family"])
    d = dict(rec["design"])
    specs = fam.fit_specs(rec["mating"], d)
    crit = sorted({s.param for s in specs if s.param in d})
    if kind in ("tight", "loose", "nudge") and crit:
        p = crit[int(rng.integers(len(crit)))]
        s = next(s for s in specs if s.param == p)
        mag = rng.uniform(0.02, 0.12) if kind == "nudge" else rng.uniform(0.15, 1.0)
        sign = (-1 if kind == "tight" else 1) * s.gap_sign
        if kind == "nudge":
            sign = rng.choice([-1, 1])
        d[p] = round(d[p] + sign * mag / max(s.param_scale, 1e-3) * 0.5, 3)
    elif kind == "rescale":
        keys = [k for k, v in d.items() if isinstance(v, float) and v > 0.5 and k not in crit]
        for k in rng.choice(keys, size=min(len(keys), int(rng.integers(1, 4))), replace=False) if keys else []:
            d[k] = round(d[k] * rng.uniform(0.5, 1.5), 3)
    return d


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default="results/m2/verifier_agreement.json")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    recs = []
    for name in ("test_id", "val", "test"):
        recs += read_jsonl(f"data/benchmarks/{name}.jsonl")
    pick = rng.choice(len(recs), size=min(a.n, len(recs)), replace=False)
    kinds = ["oracle", "tight", "loose", "nudge", "rescale"]
    items = []
    for i in pick:
        r = recs[int(i)]
        fam = get_family(r["family"])
        for k in kinds:
            d = r["design"] if k == "oracle" else mutate(r, rng, k)
            items.append({"uid": r["uid"], "family": r["family"], "kind": k, "rec": r,
                          "src": fam.program(d, r["style"])})
    t0 = time.time()
    with VerifyPool(a.workers, timeout=180) as pool:
        prim = pool.run([VerifyJob(it["src"], it["rec"], "primary") for it in items])
        held = pool.run([VerifyJob(it["src"], it["rec"], "heldout") for it in items])
    secs = time.time() - t0
    rows = []
    for it, p, h in zip(items, prim, held):
        rows.append({"uid": it["uid"], "family": it["family"], "kind": it["kind"],
                     **{f"p_{c}": bool(p.get(c, False)) for c in COMPONENTS},
                     **{f"h_{c}": bool(h.get(c, False)) for c in COMPONENTS},
                     "p_reward": p.get("reward", 0.0), "h_reward": h.get("reward", 0.0),
                     "h_err": h.get("error_kind", ""), "p_err": p.get("error_kind", "")})
    uids = sorted({r["uid"] for r in rows})
    summary = {}
    for c in COMPONENTS:
        pa = np.array([r[f"p_{c}"] for r in rows])
        ha = np.array([r[f"h_{c}"] for r in rows])
        per_part = [np.mean([r[f"p_{c}"] == r[f"h_{c}"] for r in rows if r["uid"] == u]) for u in uids]
        pt, lo, hi = bootstrap_ci(per_part, seed=a.seed)
        summary[c] = {"agreement": pt, "ci95": [lo, hi], "kappa": cohen_kappa(pa, ha),
                      "primary_pass_rate": float(pa.mean()), "heldout_pass_rate": float(ha.mean()),
                      "primary_only": int((pa & ~ha).sum()), "heldout_only": int((~pa & ha).sum()),
                      "meets_threshold": bool(pt >= THRESHOLD)}
    by_family = {}
    for f in sorted({r["family"] for r in rows}):
        rr = [r for r in rows if r["family"] == f]
        by_family[f] = {c: float(np.mean([r[f"p_{c}"] == r[f"h_{c}"] for r in rr])) for c in COMPONENTS}
    pr = np.array([r["p_reward"] for r in rows])
    hr = np.array([r["h_reward"] for r in rows])
    payload = {"n_parts": len(uids), "n_programs": len(rows), "threshold": THRESHOLD, "seconds": round(secs, 1),
               "summary": summary, "by_family": by_family,
               "reward_spearman": float(np.corrcoef(np.argsort(np.argsort(pr)), np.argsort(np.argsort(hr)))[0, 1]),
               "rows": rows}
    write_result(a.out, payload, stamp(a.seed, vars(a)))
    for c, s in summary.items():
        print(f"{c:14s} agree={s['agreement']:.3f} [{s['ci95'][0]:.3f},{s['ci95'][1]:.3f}] kappa={s['kappa']:.2f} "
              f"P={s['primary_pass_rate']:.2f} H={s['heldout_pass_rate']:.2f} P-only={s['primary_only']} H-only={s['heldout_only']}")


if __name__ == "__main__":
    main()
