"""Fit-critical feature inference: measurement fusion, mating inference, standard-size priors, and the
tolerance engine, producing the structured representation (feature graph + dims + uncertainty).

Two uses:
  * real parts: ``infer_from_observation`` takes caliper values / metrology / hints for a part type;
  * synthetic benchmark: ``observe`` simulates the measurement channel for a frozen record, using the
    *empirical* photo-metrology residuals measured in M1 (results/m1/metrology_synth.json) and a caliper
    noise model, so generators are evaluated on realistically noisy inputs and the ablations
    (-tolerance engine, -mating inference, -uncertainty/standard sizes) are well defined.

Structure-vs-data rule: which part parameter mates with which counterpart parameter is part of the
part-type schema (the feature graph's ``mates_with``) and is read from the family definition; any
*numeric prior* (molded clearance offsets) is learned only from train-family records.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import cache, lru_cache
from pathlib import Path

import numpy as np

from spare.features.tolerance import DEFAULT_PRINTER, PrinterProfile, printed_deviation
from spare.repr.render import render
from spare.synth.registry import FAMILIES, families_in

ROOT = Path(__file__).resolve().parents[2]
CALIPER_SIGMA = 0.03  # mm, digital caliper on plastic incl. operator error (assumption, D-017)

# Standard hardware sizes (mm). Real shafts/screws are nominal or slightly undersize.
STANDARD_SIZES: dict[str, list[float]] = {
    "screw": [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0],
    "bolt": [2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0],
    "shaft": [3.0, 4.0, 5.0, 6.0, 6.35, 8.0],
    "axle": [2.0, 3.0, 4.0, 5.0, 6.0, 8.0],
    "pin": [1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
}
COUNT_PARAMS = {"grip_ribs", "hex", "base_holes", "upright_holes"}
CALIPER_WORDS = ("diameter", "flat", "thickness", "wall", "aperture")


def standard_kind(param: str) -> str | None:
    for k in STANDARD_SIZES:
        if param.startswith(k + "_diameter"):
            return k
    return None


def snap_standard(x: float, sigma: float, kind: str | None, undersize: float = 0.03) -> tuple[float, bool]:
    """Snap to a standard size if one lies within max(2 sigma, 0.12 mm); returns (value, snapped)."""
    if kind is None:
        return x, False
    sizes = np.array(STANDARD_SIZES[kind])
    k = int(np.argmin(np.abs(sizes - x)))
    if abs(sizes[k] - x) <= max(2 * sigma, 0.12):
        return float(sizes[k] - undersize), True
    return x, False


# ----------------------------------------------------------------------------------------- links/priors
@cache
def links(family: str, n: int = 150, seed: int = 0) -> dict[str, tuple[str, str, str]]:
    """{original_param: (mating_param, fit_class, kind)} for fit-critical params, from the part-type schema:
    the fit spec says which design param controls the gap; the counterpart param is the mating dim that
    co-varies with it in the type's geometry (structure, not data)."""
    fam = FAMILIES[family]
    rng = np.random.default_rng(seed)
    rows = [fam.instance(rng, f"l{i}") for i in range(n)]
    specs = fam.fit_specs(rows[0].mating, rows[0].design)
    out = {}
    for s in specs:
        p = s.param
        if p not in rows[0].original:
            continue
        xo = np.array([r.original[p] for r in rows])
        best, bc = None, 0.0
        for q in rows[0].mating:
            xm = np.array([r.mating[q] for r in rows])
            if xm.std() < 1e-9 or xo.std() < 1e-9:
                continue
            c = abs(np.corrcoef(xo, xm)[0, 1])
            if c > bc:
                best, bc = q, c
        if best is not None and bc > 0.9:
            out[p] = (best, s.fit_class, s.kind)
    return out


@lru_cache(maxsize=1)
def clearance_priors() -> dict[str, float]:
    """Median molded offset (original part param - mating param) per fit class, learned from *train-family*
    records only (mating inference when the counterpart is not available)."""
    from spare.common.provenance import read_jsonl

    p = ROOT / "data" / "benchmarks" / "test_id.jsonl"  # held-out instances of TRAIN families
    recs = read_jsonl(p) if p.exists() else []
    if not recs:
        rng = np.random.default_rng(0)
        recs = []
        for f in families_in("train"):
            for _i in range(30):
                inst = f.instance(rng, "p")
                recs.append({"family": f.name, "mating": inst.mating, "original": inst.original})
    by: dict[str, list[float]] = {}
    for r in recs:
        if FAMILIES[r["family"]].split != "train":
            continue
        for p_, (q, fc, kind) in links(r["family"]).items():
            off = r["original"][p_] - r["mating"][q]
            if abs(off) < 3.0:  # only same-quantity links (diameter-to-diameter etc.)
                by.setdefault(f"{fc}:{kind}", []).append(off)
                by.setdefault(fc, []).append(off)
    return {k: float(np.median(v)) for k, v in by.items()}


@lru_cache(maxsize=1)
def photo_residuals() -> dict[str, np.ndarray]:
    """Signed photo-metrology residuals (meas - true, mm) by feature type from M1."""
    p = ROOT / "results" / "m1" / "metrology_synth.json"
    out: dict[str, list[float]] = {}
    if p.exists():
        d = json.loads(p.read_text())
        for res in d["parts"]:
            for r in res.get("rows", []):
                if r.get("meas") is not None:
                    out.setdefault(r["feature"], []).append(r["meas"] - r["true"])
    defaults = {"extent": [0.6, -0.4, 0.9, -0.2], "height": [0.2, -1.5], "hole_d": [1.2, -1.0]}
    return {k: np.array(out.get(k) or defaults[k]) for k in ("extent", "height", "hole_d")}


@lru_cache(maxsize=1)
def sigma_scales() -> dict[str, float]:
    p = ROOT / "spare" / "metrology" / "sigma_scale.json"
    return json.loads(p.read_text()) if p.exists() else {}


# ----------------------------------------------------------------------------------------- observation
@dataclass
class Observation:
    mating: dict[str, float]
    original: dict[str, float]
    sigma: dict[str, float] = field(default_factory=dict)  # per param (mm)
    source: dict[str, str] = field(default_factory=dict)  # caliper | photo | inferred | standard | exact | user
    notes: list[str] = field(default_factory=list)


CONDITIONS = ("caliper", "photo_only", "no_mating_item")


@cache
def critical_params(family: str, seed: int = 0) -> frozenset[str]:
    """Fit/strength-critical original params, found structurally: the fit-spec params, any param whose change
    moves a fit-spec sample point on the mating surface (positions such as hole spacing), critical probes,
    and thickness/wall/diameter-like dims. These are what the capture guide asks the user to caliper."""
    fam = FAMILIES[family]
    inst = fam.instance(np.random.default_rng(seed), "c")
    base = fam.fit_specs(inst.mating, inst.design)
    crit = {sp.param for sp in base} | {p.name for p in fam.probes(inst.original) if p.critical}
    for k, v in inst.design.items():
        if k in COUNT_PARAMS:
            continue
        d2 = dict(inst.design)
        d2[k] = v + (0.05 if 0 < abs(v) < 1 else 0.5)
        try:
            sp2 = fam.fit_specs(inst.mating, d2)
        except Exception:  # noqa: BLE001
            continue
        if any(a.points.shape != b.points.shape or np.abs(a.points - b.points).max() > 1e-6
               for a, b in zip(base, sp2)):
            crit.add(k)
    crit |= {k for k in inst.original if any(w in k for w in CALIPER_WORDS)}
    return frozenset(crit & set(inst.original))


def _photo(x: float, rng, resid: np.ndarray, fraction: bool = False) -> float:
    e = float(rng.choice(resid))
    if fraction:  # dimensionless position fraction: image error relative to a ~20 mm reference length
        return float(np.clip(x + e / 20.0, 0.05, 0.95))
    e = float(np.clip(e, -0.15 * abs(x), 0.15 * abs(x))) if abs(x) < 10 else e  # small dims: feature-relative
    return x + e


def observe(rec: dict, cond: str, rng: np.random.Generator, standard_sizes: bool = True) -> Observation:
    """Simulate what the pipeline observes for a frozen record under a measurement condition."""
    res = photo_residuals()
    crit = critical_params(rec["family"])
    lk = links(rec["family"])
    ob = Observation({}, {})
    for k, v in rec["original"].items():
        if k in COUNT_PARAMS or (float(v).is_integer() and k not in crit and abs(v) < 100 and "diameter" not in k
                                 and k in ("hex",)):
            ob.original[k], ob.source[k] = v, "exact"
            continue
        if k in COUNT_PARAMS:
            continue
        frac = 0 < v < 1 and ("fraction" in k or k.endswith("_x"))
        if cond != "photo_only" and k in crit:
            ob.original[k] = round(v + rng.normal(0, CALIPER_SIGMA / (20.0 if frac else 1.0)), 4)
            ob.sigma[k], ob.source[k] = CALIPER_SIGMA, "caliper"
        else:
            r = res["hole_d"] if "diameter" in k and k in crit else res["extent"]
            ob.original[k] = round(_photo(v, rng, r, fraction=frac), 4)
            ob.sigma[k], ob.source[k] = float(np.median(np.abs(r)) / 0.674), "photo"
    pri = clearance_priors()
    inv = {q: (p, fc, kind) for p, (q, fc, kind) in lk.items()}
    for q, v in rec["mating"].items():
        if q == "load_N":
            ob.mating[q], ob.source["m:" + q] = v, "user"
            continue
        if cond == "no_mating_item" and q in inv:
            p, fc, kind = inv[q]
            off = pri.get(f"{fc}:{kind}", pri.get(fc, 0.0))
            x, s = ob.original[p] - off, max(ob.sigma.get(p, 0.3), 0.08)
            src = "inferred"
        elif cond == "photo_only":
            r = res["hole_d"] if "diameter" in q else res["extent"]
            x, s, src = _photo(v, rng, r), float(np.median(np.abs(r)) / 0.674), "photo"
        else:
            x, s, src = v + rng.normal(0, CALIPER_SIGMA), CALIPER_SIGMA, "caliper"
        if standard_sizes:
            x2, snapped = snap_standard(x, s, standard_kind(q))
            if snapped:
                x, s, src = x2, 0.03, src + "+standard"
        ob.mating[q] = round(float(x), 3)
        ob.sigma["m:" + q], ob.source["m:" + q] = s, src
    return ob


# ----------------------------------------------------------------------------------------- design
DESIGN_MODES = ("full", "no_tolerance", "no_mating")


def infer_design(rec: dict, ob: Observation, mode: str = "full", printer: PrinterProfile = DEFAULT_PRINTER) -> dict:
    """Design dims from an observation.
    full         -- tolerance engine on inferred mating dims (+ DfAM rules)
    no_tolerance -- copy the measured original (geometry-similarity)
    no_mating    -- reproduce the measured original *as printed*: compensate printer deviation on fit params
                    but keep the original's (molded) clearances, i.e. no counterpart reasoning."""
    fam = FAMILIES[rec["family"]]
    if mode == "full":
        return fam.derive_design(ob.mating, ob.original, printer, rec.get("material", "PLA"))
    d = dict(ob.original)
    if mode == "no_mating":
        for s in fam.fit_specs(ob.mating, d):
            if s.param in d:
                dev = printed_deviation(s.kind, s.feature_size, printer)
                d[s.param] = round(d[s.param] + s.gap_sign * dev / max(s.param_scale, 1e-3), 3)
    return d


def repr_for(rec: dict, ob: Observation, design: dict, with_sigma: bool = True) -> str:
    fam = FAMILIES[rec["family"]]
    g = fam.feature_graph(ob.mating, design, ob.original)
    sig = None
    if with_sigma:
        sig = {}
        for f in g["features"]:
            # feature uncertainty = max sigma of params named in the feature (approximate)
            vals = [ob.sigma[k] for k in ob.sigma if not k.startswith("m:") and k.split("_")[0] in f["id"]]
            if vals:
                sig[f["id"]] = max(vals)
    mat = rec.get("material", "PLA")
    return render(g, mat, "generic_petg_0.4" if mat == "PETG" else "generic_pla_0.4", sig)


def infer_record(rec: dict, cond: str = "caliper", mode: str = "full", seed: int = 0,
                 standard_sizes: bool = True, with_sigma: bool = True) -> dict:
    """Noisy-input benchmark record: same uid/task, generator input built from simulated observation."""
    rng = np.random.default_rng((hash(rec["uid"]) & 0xFFFF_FFFF) ^ (seed * 7919))
    ob = observe(rec, cond, rng, standard_sizes)
    d = infer_design(rec, ob, mode)
    return {**{k: rec[k] for k in ("uid", "family", "split", "style", "material")}, "cond": cond, "mode": mode,
            "obs_mating": ob.mating, "obs_original": ob.original, "obs_sigma": ob.sigma, "obs_source": ob.source,
            "design_hat": d, "repr": repr_for(rec, ob, d, with_sigma)}


_ = math
