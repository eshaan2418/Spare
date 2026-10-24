"""Simulated fit verifier.

For each FitSpec we measure the signed directional gap from sampled points on the
*mating* functional surface to the generated part (design geometry), then apply
the printer deviation model to get the *printed* gap, and test it against the
fit-class band.  A global interference check (part ∩ mating volume) catches
parts that fill the mating space elsewhere.

Primary backend: OCCT B-rep classification + ray casting + boolean common.
Held-out backend (verifiers/heldout): triangle-mesh ray casting at a different
tessellation, jittered sample points, and manifold3d booleans.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from spare.common.geom import (
    MeshQuery,
    OCCTQuery,
    shape_to_mesh,
    signed_directional_gap_mesh,
    signed_directional_gap_occt,
)
from spare.features.tolerance import DEFAULT_PRINTER, FIT_BANDS, PrinterProfile, printed_deviation
from spare.synth.family import FitSpec

MIN_IN_BAND = 0.85     # fraction of functional-surface samples whose printed gap is in band
BIND_MARGIN = 0.10     # any sample tighter than lo - margin = local binding -> fail
MAX_MISSING = 0.05     # fraction of samples with no part surface within search range


@dataclass
class SpecResult:
    name: str
    fit_class: str
    location: str
    param: str
    passed: bool
    verdict: str  # ok | too_tight | too_loose | missing | binding
    frac_in_band: float
    median_printed_gap: float
    p10: float
    p90: float
    frac_missing: float
    band: tuple[float, float]
    # signed per-side correction (mm) that would centre the median in band
    suggested_gap_change: float


@dataclass
class FitResult:
    passed: bool
    specs: list[SpecResult]
    interference_volume: float = 0.0
    interference_allowed: float = 0.0
    interference_ok: bool = True
    backend: str = "occt"
    notes: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        """Smooth reward in [0,1]: mean in-band fraction, zeroed by gross interference."""
        if not self.specs:
            return 0.0
        s = float(np.mean([sr.frac_in_band for sr in self.specs]))
        return s if self.interference_ok else 0.25 * s


def _judge(spec: FitSpec, gaps_design: np.ndarray, pr: PrinterProfile) -> SpecResult:
    lo, hi = FIT_BANDS[spec.fit_class]
    dev = printed_deviation(spec.kind, spec.feature_size, pr)
    missing = ~np.isfinite(gaps_design)
    g = gaps_design[~missing] - dev
    fm = float(missing.mean()) if len(gaps_design) else 1.0
    if len(g) == 0:
        return SpecResult(spec.name, spec.fit_class, spec.location, spec.param, False, "missing", 0.0,
                          float("nan"), float("nan"), float("nan"), 1.0, (lo, hi), 0.0)
    inb = (g >= lo) & (g <= hi)
    frac = float(inb.sum() / len(gaps_design))
    med = float(np.median(g))
    p10, p90 = float(np.percentile(g, 10)), float(np.percentile(g, 90))
    binding = bool((g < lo - BIND_MARGIN).mean() > 0.02)
    passed = frac >= MIN_IN_BAND and fm <= MAX_MISSING and not binding
    if passed:
        verdict = "ok"
    elif fm > MAX_MISSING and fm >= 0.5:
        verdict = "missing"
    elif med < lo or binding:
        verdict = "too_tight"
    elif med > hi:
        verdict = "too_loose"
    else:
        verdict = "too_tight" if (g < lo).sum() >= (g > hi).sum() else "too_loose"
    center = 0.5 * (lo + hi)
    return SpecResult(spec.name, spec.fit_class, spec.location, spec.param, passed, verdict, frac, med, p10, p90,
                      fm, (lo, hi), float(center - med))


def _interference_occt(shape, mating_shape) -> float:
    try:
        common = shape.intersect(mating_shape)
        return float(max(common.Volume(), 0.0))
    except Exception:  # noqa: BLE001
        return float("nan")


def _interference_manifold(mesh, mating_mesh) -> float:
    import manifold3d as m3d

    def to_m(tm):
        return m3d.Manifold(m3d.Mesh(np.asarray(tm.vertices, np.float32), np.asarray(tm.faces, np.uint32)))

    try:
        return float(max((to_m(mesh) ^ to_m(mating_mesh)).volume(), 0.0))
    except Exception:  # noqa: BLE001
        return float("nan")


def allowed_interference(specs: list[FitSpec], pr: PrinterProfile) -> float:
    """Press fits legitimately overlap the nominal mating volume: allow area * |band lo| * 2."""
    allowed = 0.5  # mm^3 numerical slack
    for s in specs:
        if FIT_BANDS[s.fit_class][0] < 0:
            area = getattr(s, "area", 0.0) or _approx_area(s)
            allowed += area * (abs(FIT_BANDS[s.fit_class][0]) + 0.05) * 1.5
    return allowed


def _approx_area(s: FitSpec) -> float:
    p = s.points
    if len(p) < 3:
        return 0.0
    ext = np.ptp(p, axis=0)
    e = np.sort(ext)[::-1]
    # cylinder-ish sample sets: perimeter ~ pi * diameter
    return float(np.pi * e[0] * e[2]) if e[1] > 0.5 * e[0] else float(e[0] * e[1])


def check_fit(shape, specs: list[FitSpec], mating_shape=None, pr: PrinterProfile = DEFAULT_PRINTER,
              backend: str = "occt", search: float = 5.0, mesh=None, mating_mesh=None, seed: int = 0) -> FitResult:
    results = []
    if backend == "occt":
        q = OCCTQuery(shape)
        for s in specs:
            gaps = signed_directional_gap_occt(q, s.points, s.normals, search)
            results.append(_judge(s, gaps, pr))
    else:
        mesh = mesh if mesh is not None else shape_to_mesh(shape, tol=0.01, ang_tol=0.1)
        mq = MeshQuery(mesh)
        rng = np.random.default_rng(seed + 7919)
        for s in specs:
            pts = _jitter(s.points, s.normals, rng)
            gaps = signed_directional_gap_mesh(mq, pts, s.normals, search)
            results.append(_judge(s, gaps, pr))
    inter, allowed, inter_ok = 0.0, allowed_interference(specs, pr), True
    if mating_shape is not None:
        if backend == "occt":
            inter = _interference_occt(shape, mating_shape)
        else:
            mm = mating_mesh if mating_mesh is not None else shape_to_mesh(mating_shape, 0.01, 0.1)
            inter = _interference_manifold(mesh, mm)
        inter_ok = bool(np.isfinite(inter) and inter <= allowed)
    passed = bool(results) and all(r.passed for r in results) and inter_ok
    return FitResult(passed, results, inter, allowed, inter_ok, backend)


def _jitter(pts: np.ndarray, nrm: np.ndarray, rng, amp: float = 0.15) -> np.ndarray:
    """Tangential jitter of sample points (held-out uses different sample locations)."""
    r = rng.normal(size=pts.shape)
    r -= (r * nrm).sum(1, keepdims=True) * nrm
    return pts + amp * r
