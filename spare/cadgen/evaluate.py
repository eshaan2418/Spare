"""Shared evaluation harness for every program generator (M4 baselines, M5 SFT/RL, ablations).

Protocol (DECISIONS D-017, D-019, D-020):
  1. Each frozen record is *observed* through the simulated measurement channel (calipers on critical dims,
     photos elsewhere). Two inputs can be built from one observation:
       ``spare``      -- mating inference + tolerance engine (function-first design dims)
       ``similarity`` -- the measured original geometry copied (geometry-similarity)
  2. A generator returns N candidate programs per input (equal budget for every system).
  3. Selection: ``first`` takes candidate 0; ``best_of_n`` runs the *primary* verifier on the *inferred* task
     (observed mating dims, default printer) and picks the highest reward. No ground truth is used.
  4. The selected program is scored by the *held-out* suite against the TRUE mating geometry under the part's
     hidden virtual printer (fit@1). One correction from the simulated categorical fit report (the policy only
     sees outcome + location) gives attempt 2 (fit@2).
Every reported number is `simulated`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from spare.common.provenance import bootstrap_ci, paired_bootstrap_diff
from spare.features.infer import infer_record
from spare.feedback.policy import correct, simulated_report
from spare.synth.registry import get as get_family
from spare.verifiers.pool import VerifyJob, VerifyPool
from spare.verifiers.suite import virtual_printer_for

INPUT_MODES = {"spare": "full", "similarity": "no_tolerance"}
HELDOUT_FULL: dict = {}
HELDOUT_FAST = {"slicer": False, "fea": False, "functional_edit": False}
PRIMARY_OPTS: dict = {}

# A generator maps (records, observed inputs, n, seed) -> n programs per record.
Generator = Callable[[list[dict], list[dict], int, int], list[list[str]]]


@dataclass
class System:
    name: str
    input: str  # spare | similarity
    gen: Generator
    select: str = "first"  # first | best_of_n
    attempts: int = 2


def observe_all(recs: list[dict], input_mode: str, seed: int = 0, cond: str = "caliper") -> list[dict]:
    return [infer_record(r, cond, INPUT_MODES[input_mode], seed) for r in recs]


def inferred_record(rec: dict, x: dict) -> dict:
    """The task the pipeline can build at inference time: observed mating dims + its own design dims."""
    return {**rec, "mating": x["obs_mating"], "design": x["design_hat"], "original": x["obs_original"]}


def template_generator(use: str = "design_hat") -> Generator:
    """Oracle program generator: the family template on the inferred dims (or on the exact original)."""
    def gen(recs, xs, n, seed):
        out = []
        for r, x in zip(recs, xs):
            p = r["original"] if use == "original" else x["design_hat"]
            out.append([get_family(r["family"]).program(p, r["style"])] * n)
        return out
    return gen


def run_system(sys_: System, recs: list[dict], pool: VerifyPool, n: int = 8, seed: int = 0,
               heldout_opts: dict | None = None, log=print) -> list[dict]:
    xs = observe_all(recs, sys_.input, seed)
    # first-sample systems need one candidate; best-of-N gets the full budget (cached samples are shared)
    cands = sys_.gen(recs, xs, n if sys_.select == "best_of_n" else 1, seed)
    assert len(cands) == len(recs)
    # ---- selection (primary verifier, inferred task only)
    if sys_.select == "best_of_n" and n > 1:
        jobs = [VerifyJob(c, inferred_record(r, x), "primary", None, opts=PRIMARY_OPTS)
                for r, x, cs in zip(recs, xs, cands) for c in cs]
        reps = pool.run(jobs)
        sel, k = [], 0
        for cs in cands:
            rw = [reps[k + i].get("reward", 0.0) for i in range(len(cs))]
            sel.append(cs[int(np.argmax(rw))])
            k += len(cs)
        n_primary = len(jobs)
    else:
        sel = [cs[0] for cs in cands]
        n_primary = 0
    ho = heldout_opts if heldout_opts is not None else HELDOUT_FULL
    vps = [virtual_printer_for(r["uid"]).name for r in recs]
    reps1 = pool.run([VerifyJob(s, r, "heldout", vp, opts=ho) for s, r, vp in zip(sel, recs, vps)])
    # ---- attempt 2: one correction from the categorical report of attempt 1
    srcs2, notes = [], []
    for s, r, x, rep in zip(sel, recs, xs, reps1):
        if not rep.get("ok_exec") or not rep.get("valid") or (rep.get("fit_pass") and rep.get("strength_pass")):
            srcs2.append(None)
            notes.append("" if rep.get("ok_exec") else "no program to correct")
            continue
        fr = simulated_report(r["uid"], 1, rep.get("fit_detail", []), bool(rep.get("strength_pass", True)))
        fam = get_family(r["family"])
        specs = fam.fit_specs(x["obs_mating"], x["design_hat"])
        c = correct(s, fr, specs)
        srcs2.append(c.src if c.applied else None)
        notes.append(f"{fr.outcome}@{fr.location}: " + (", ".join(f"{e.param} {e.old}->{e.new}" for e in c.edits)
                                                         if c.applied else c.note))
    idx2 = [i for i, s in enumerate(srcs2) if s is not None]
    reps2 = pool.run([VerifyJob(srcs2[i], recs[i], "heldout", vps[i], opts=ho) for i in idx2]) if idx2 else []
    r2 = dict(zip(idx2, reps2))
    rows = []
    for i, (r, rep) in enumerate(zip(recs, reps1)):
        ok1 = bool(rep.get("fit_pass")) and bool(rep.get("valid"))
        rep2 = r2.get(i)
        ok2 = ok1 or (rep2 is not None and bool(rep2.get("fit_pass")) and bool(rep2.get("valid")))
        func1 = ok1 and bool(rep.get("strength_pass"))
        func2 = func1 or (rep2 is not None and bool(rep2.get("fit_pass")) and bool(rep2.get("valid"))
                          and bool(rep2.get("strength_pass")))
        rows.append({
            "system": sys_.name, "uid": r["uid"], "family": r["family"], "split": r["split"], "printer": vps[i],
            "exec": bool(rep.get("ok_exec")), "valid": bool(rep.get("valid")), "error_kind": rep.get("error_kind", ""),
            "fit1": ok1, "fit2": ok2, "func1": func1, "func2": func2,
            "print_pass": bool(rep.get("print_pass")), "strength_pass": bool(rep.get("strength_pass")),
            "dims_pass": bool(rep.get("dims_pass")), "dims_score": float(rep.get("dims_score") or 0.0),
            "edit_score": float(rep.get("edit_score") or 0.0), "edit_pass": bool(rep.get("edit_pass")),
            "all_pass1": bool(rep.get("all_pass")),
            "all_pass2": bool(rep.get("all_pass")) or (rep2 is not None and bool(rep2.get("all_pass"))),
            "correction": notes[i], "n_primary_calls": n_primary / max(len(recs), 1),
            "program": sel[i],
            "fit_verdicts": [f.get("verdict") for f in rep.get("fit_detail", []) if "verdict" in f],
            "error": (rep.get("error") or "")[:200],
        })
    s = summarize(rows)
    log(f"[{sys_.name}] n={len(rows)} exec={s['exec'][0]:.2f} fit1={s['fit1'][0]:.3f} fit2={s['fit2'][0]:.3f} "
        f"all1={s['all_pass1'][0]:.3f} print={s['print_pass'][0]:.2f} str={s['strength_pass'][0]:.2f} "
        f"edit={s['edit_score'][0]:.2f}")
    return rows


METRICS = ("exec", "valid", "fit1", "fit2", "func1", "func2", "all_pass1", "all_pass2", "print_pass",
           "strength_pass", "dims_pass", "edit_score", "edit_pass")


def summarize(rows: list[dict], seed: int = 0) -> dict:
    out = {}
    for m in METRICS:
        v = [float(r[m]) for r in rows]
        out[m] = list(bootstrap_ci(v, seed=seed)) if v else [float("nan")] * 3
    out["n"] = len(rows)
    return out


def summarize_by(rows: list[dict], key: str = "split", seed: int = 0) -> dict:
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r[key], []).append(r)
    return {k: summarize(v, seed) for k, v in sorted(groups.items())}


def paired(rows_a: list[dict], rows_b: list[dict], metric: str = "fit1", seed: int = 0) -> dict:
    a = {r["uid"]: float(r[metric]) for r in rows_a}
    b = {r["uid"]: float(r[metric]) for r in rows_b}
    u = sorted(set(a) & set(b))
    if not u:
        return {}
    d, lo, hi, p = paired_bootstrap_diff([a[k] for k in u], [b[k] for k in u], seed=seed)
    return {"n": len(u), "diff": d, "ci95": [lo, hi], "p": p}
