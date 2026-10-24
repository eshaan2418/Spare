"""Geometric probes: measure dimensions on an arbitrary solid (never read program parameters).

Kinds:
  extent        {axis}                      bbox size along axis
  extent_max    {axis}                      bbox max coordinate along axis
  circle_d      {axis, offset, near, which} diameter of the innermost/outermost loop around `near`
                                            in the section plane (robust circle fit)
  ray           {origin, dir}               distance to the first surface hit
  thickness     {origin, dir}               material thickness: first entry -> next exit along ray
  depth_from_top{at}                        top of bbox minus first hit going down at (x, y)

Primary backend: OCCT (section + IntCurvesFace).  Held-out backend: trimesh.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from spare.synth.family import Probe

INF = float("inf")


@dataclass
class ProbeResult:
    name: str
    measured: float
    target: float
    tol: float
    critical: bool
    ok: bool

    @property
    def err(self) -> float:
        return abs(self.measured - self.target) if math.isfinite(self.measured) else INF


def _robust_circle(pts2: np.ndarray) -> tuple[float, float, float]:
    """Algebraic (Kasa) circle fit with 2 rounds of residual-based outlier rejection."""
    p = np.asarray(pts2, float)
    if len(p) < 3:
        return float("nan"), float("nan"), float("nan")
    keep = np.ones(len(p), bool)
    cx = cy = r = float("nan")
    for _ in range(3):
        q = p[keep]
        A = np.c_[2 * q[:, 0], 2 * q[:, 1], np.ones(len(q))]
        b = (q ** 2).sum(1)
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        cx, cy = sol[0], sol[1]
        r = math.sqrt(max(sol[2] + cx * cx + cy * cy, 0.0))
        res = np.abs(np.hypot(p[:, 0] - cx, p[:, 1] - cy) - r)
        mad = np.median(res[keep]) + 1e-6
        new = res < max(4 * mad, 0.02)
        if new.sum() < 6 or (new == keep).all():
            break
        keep = new
    return float(cx), float(cy), float(r)


def _loop_points(pts2: np.ndarray, near, which: str, nbins: int = 90) -> np.ndarray:
    """Pick the innermost ('inner') or outermost ('outer') boundary around `near` by polar binning."""
    if len(pts2) == 0:
        return pts2
    d = pts2 - np.asarray(near, float)[None]
    ang = np.arctan2(d[:, 1], d[:, 0])
    rad = np.hypot(d[:, 0], d[:, 1])
    bins = ((ang + np.pi) / (2 * np.pi) * nbins).astype(int) % nbins
    out = []
    for b in range(nbins):
        m = bins == b
        if not m.any():
            continue
        idx = np.where(m)[0]
        j = idx[np.argmin(rad[idx])] if which == "inner" else idx[np.argmax(rad[idx])]
        out.append(pts2[j])
    return np.array(out)


def _axes(axis: int) -> tuple[int, int]:
    return [(1, 2), (0, 2), (0, 1)][axis]


# ------------------------------------------------------------------ OCCT backend
def _occt_section_points(shape, axis: int, offset: float, step: float = 0.05) -> np.ndarray:
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Section
    from OCP.GCPnts import GCPnts_UniformAbscissa
    from OCP.gp import gp_Dir, gp_Pln, gp_Pnt
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    n = [0.0, 0.0, 0.0]
    n[axis] = 1.0
    o = [0.0, 0.0, 0.0]
    o[axis] = offset
    sec = BRepAlgoAPI_Section(shape.wrapped, gp_Pln(gp_Pnt(*o), gp_Dir(*n)))
    sec.Build()
    pts = []
    exp = TopExp_Explorer(sec.Shape(), TopAbs_EDGE)
    while exp.More():
        e = TopoDS.Edge_s(exp.Current())
        c = BRepAdaptor_Curve(e)
        disc = GCPnts_UniformAbscissa(c, step)
        if disc.IsDone() and disc.NbPoints() > 1:
            for i in range(1, disc.NbPoints() + 1):
                p = c.Value(disc.Parameter(i))
                pts.append((p.X(), p.Y(), p.Z()))
        else:
            for t in np.linspace(c.FirstParameter(), c.LastParameter(), 8):
                p = c.Value(t)
                pts.append((p.X(), p.Y(), p.Z()))
        exp.Next()
    a, b = _axes(axis)
    P = np.array(pts) if pts else np.zeros((0, 3))
    return P[:, [a, b]] if len(P) else np.zeros((0, 2))


class OCCTProber:
    def __init__(self, shape):
        from spare.common.geom import OCCTQuery, bbox

        self.shape = shape
        self.q = OCCTQuery(shape)
        self.lo, self.hi = bbox(shape)

    def hits(self, origin, d, tmax=1e3) -> list[float]:
        from OCP.gp import gp_Dir, gp_Lin, gp_Pnt

        lin = gp_Lin(gp_Pnt(*map(float, origin)), gp_Dir(*map(float, d)))
        self.q._isect.Perform(lin, 0.0, tmax)
        if not self.q._isect.IsDone():
            return []
        return sorted(self.q._isect.WParameter(i) for i in range(1, self.q._isect.NbPnt() + 1))

    def section(self, axis, offset):
        return _occt_section_points(self.shape, axis, offset)


class MeshProber:
    def __init__(self, mesh):
        from spare.common.geom import MeshQuery

        self.mesh = mesh
        self.mq = MeshQuery(mesh)
        self.lo, self.hi = mesh.bounds[0], mesh.bounds[1]

    def hits(self, origin, d, tmax=1e3) -> list[float]:
        o = np.asarray(origin, float)[None]
        dd = np.asarray(d, float)[None]
        locs, idx, _ = self.mq._ray.intersects_location(o, dd, multiple_hits=True)
        if len(locs) == 0:
            return []
        t = (locs - o) @ dd[0]
        t = np.unique(np.round(t[(t > 1e-9) & (t <= tmax)], 6))
        return sorted(t.tolist())

    def section(self, axis, offset):
        n = np.zeros(3)
        n[axis] = 1
        o = np.zeros(3)
        o[axis] = offset
        s = self.mesh.section(plane_origin=o, plane_normal=n)
        if s is None:
            return np.zeros((0, 2))
        P = np.asarray(s.vertices)
        # densify segments
        segs = []
        for ent in s.entities:
            pts = P[ent.points]
            for a_, b_ in zip(pts[:-1], pts[1:]):
                k = max(2, int(np.linalg.norm(b_ - a_) / 0.05))
                segs.append(np.linspace(a_, b_, k))
        P = np.vstack(segs) if segs else P
        a, b = _axes(axis)
        return P[:, [a, b]]


def measure(prober, probe: Probe) -> float:
    k, a = probe.kind, probe.args
    try:
        if k == "extent":
            return float(prober.hi[a["axis"]] - prober.lo[a["axis"]])
        if k == "extent_max":
            return float(prober.hi[a["axis"]])
        if k == "ray":
            h = prober.hits(a["origin"], a["dir"])
            return float(h[0]) if h else INF
        if k == "thickness":
            h = prober.hits(a["origin"], a["dir"])
            return float(h[1] - h[0]) if len(h) >= 2 else INF
        if k == "depth_from_top":
            x, y = a["at"]
            top = float(prober.hi[2])
            h = prober.hits((x, y, top + 1.0), (0, 0, -1))
            return float(h[0] - 1.0) if h else INF
        if k == "circle_d":
            pts = prober.section(a["axis"], a["offset"])
            if len(pts) < 6:
                return INF
            loop = _loop_points(pts, a.get("near", (0, 0)), a.get("which", "inner"))
            _, _, r = _robust_circle(loop)
            return 2 * r if math.isfinite(r) else INF
    except Exception:  # noqa: BLE001
        return INF
    raise ValueError(f"unknown probe kind {k}")


def run_probes(prober, probes: list[Probe]) -> list[ProbeResult]:
    out = []
    for p in probes:
        v = measure(prober, p)
        ok = math.isfinite(v) and abs(v - p.target) <= p.tol
        out.append(ProbeResult(p.name, v, p.target, p.tol, p.critical, ok))
    return out


def dims_score(results: list[ProbeResult]) -> float:
    """Reward in [0,1]: critical probes weigh 2x; smooth falloff beyond tolerance."""
    if not results:
        return 1.0
    num = den = 0.0
    for r in results:
        w = 2.0 if r.critical else 1.0
        e = r.err
        s = 1.0 if e <= r.tol else (0.0 if not math.isfinite(e) else math.exp(-(e - r.tol) / max(r.tol, 0.05)))
        num += w * s
        den += w
    return num / den
