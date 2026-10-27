"""Correction policy: structured fit report -> targeted parametric edit of the program.

Rule-based core:
  * too_tight / too_loose at a location -> find the fit spec whose location matches, find the program's
    top-level parameter controlling that gap (exact name, else token overlap), and move it by a per-side
    step (learned from train-family simulated prints, else a default) in the gap-opening/closing direction;
  * broke at a location -> strength edit: thicken load-bearing members (hooks, brackets, pegs) or, for
    snap features, lower bending strain (thinner beam is NOT safe for load parts, so the location decides);
  * fits -> no change.
The policy only sees the categorical report (what a human gives), never the verifier's numeric gap.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

from spare.feedback.report import FitReport
from spare.synth.family import FitSpec
from spare.verifiers.editability.editability import _set_param, top_level_params

DEFAULT_STEP = 0.15  # mm per side (assumption; replaced by the learned table when available)
SNAP_WORDS = ("snap", "hook", "clip", "lip", "arm", "latch", "c-ring", "ring", "channel")


@dataclass
class Edit:
    param: str
    old: float
    new: float
    reason: str


@dataclass
class Correction:
    src: str
    edits: list[Edit] = field(default_factory=list)
    applied: bool = False
    note: str = ""


def _tokens(s: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", s.lower()) if t and t not in {"the", "of", "on", "at"}}


def match_spec(location: str, specs: list[FitSpec]) -> FitSpec | None:
    if not specs:
        return None
    if not location:
        return specs[0] if len(specs) == 1 else None
    lt = _tokens(location)
    best, bs = None, 0.0
    for s in specs:
        st = _tokens(s.location) | _tokens(s.name) | _tokens(s.param)
        sc = len(lt & st) / max(len(lt), 1)
        if sc > bs:
            best, bs = s, sc
    return best if bs > 0 else None


def match_param(name: str, params: dict[str, float]) -> str | None:
    if name in params:
        return name
    nt = _tokens(name)
    best, bs = None, 0.0
    for p in params:
        pt = _tokens(p)
        sc = len(nt & pt) / max(len(nt | pt), 1)
        if sc > bs:
            best, bs = p, sc
    return best if bs >= 0.5 else None


def correct(src: str, rep: FitReport, specs: list[FitSpec], steps: dict[str, float] | None = None) -> Correction:
    try:
        params = top_level_params(ast.parse(src))
    except SyntaxError:
        return Correction(src, note="unparseable program")
    if rep.outcome == "fits":
        return Correction(src, note="fits: no change")
    if rep.outcome in ("too_tight", "too_loose"):
        s = match_spec(rep.location, specs)
        if s is None:
            return Correction(src, note=f"no fit feature matches location '{rep.location}'")
        p = match_param(s.param, params)
        if p is None:
            return Correction(src, note=f"no program parameter controls '{s.param}'")
        step = (steps or {}).get(f"{s.fit_class}:{s.kind}", (steps or {}).get(s.kind, DEFAULT_STEP))
        sign = 1 if rep.outcome == "too_tight" else -1  # tight -> open the gap
        delta = sign * s.gap_sign * step / max(s.param_scale, 1e-3)
        new = round(params[p] + delta, 3)
        return Correction(_set_param(src, p, new), [Edit(p, params[p], new, f"{rep.outcome} at {s.location}")],
                          True)
    if rep.outcome == "broke":
        loc = rep.location.lower()
        snap = any(w in loc for w in SNAP_WORDS) and not any(w in loc for w in ("load", "hang"))
        edits = []
        cands = [k for k in params if any(w in k for w in ("thickness", "wall"))]
        if snap:
            # snap failure = strain too high: reduce interference (aperture up / overlap down) by 15 %
            ap_ = [k for k in params if "aperture" in k or "reach" in k or "overlap" in k]
            for k in ap_[:1]:
                f = 1.15 if "aperture" in k else 0.85
                edits.append(Edit(k, params[k], round(params[k] * f, 3), "snap broke: lower strain"))
            if not edits:
                for k in cands[:1]:
                    edits.append(Edit(k, params[k], round(params[k] * 0.85, 3), "snap broke: thinner beam"))
        else:
            for k in cands[:2]:
                edits.append(Edit(k, params[k], round(params[k] * 1.25, 3), "broke under load: thicker"))
            for k in [k for k in params if "fillet" in k][:1]:
                edits.append(Edit(k, params[k], round(params[k] * 1.5 + 0.3, 3), "broke: larger root fillet"))
        out = src
        for e in edits:
            out = _set_param(out, e.param, e.new)
        return Correction(out, edits, bool(edits), "" if edits else "no strength parameter found")
    return Correction(src, note=f"outcome {rep.outcome}: reprint with the same file and check bed adhesion")


def simulated_report(part_id: str, attempt: int, fit_detail: list[dict], strength_pass: bool = True) -> FitReport:
    """Categorical report a tester would give, from a held-out verifier result (simulated evidence)."""
    specs = [f for f in fit_detail if "verdict" in f]
    bad = [f for f in specs if not f.get("passed", f.get("verdict") == "ok")]
    if not strength_pass:
        return FitReport(part_id, attempt, "broke", "", evidence="simulated")
    if not bad:
        inter = [f for f in fit_detail if "interference" in f]
        if inter and not inter[0].get("ok", True):
            return FitReport(part_id, attempt, "too_tight", "", note="does not go on", evidence="simulated")
        return FitReport(part_id, attempt, "fits", evidence="simulated")
    f = bad[0]
    v = f["verdict"]
    outcome = "too_tight" if v in ("too_tight", "binding") else "too_loose" if v in ("too_loose", "missing") else "too_tight"
    return FitReport(part_id, attempt, outcome, f.get("location", ""), evidence="simulated")
