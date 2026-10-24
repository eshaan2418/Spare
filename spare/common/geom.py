"""Geometry helpers shared by verifiers: tessellation, OCCT queries, mesh queries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def shape_to_mesh(shape, tol: float = 0.02, ang_tol: float = 0.2):
    """Tessellate a cq.Shape into a trimesh.Trimesh."""
    import trimesh

    verts, tris = shape.tessellate(tol, ang_tol)
    v = np.array([(p.x, p.y, p.z) for p in verts], dtype=float)
    f = np.array(tris, dtype=np.int64).reshape(-1, 3)
    m = trimesh.Trimesh(v, f, process=True)
    return m


def solids_of(shape) -> list:
    return list(shape.Solids())


@dataclass
class Validity:
    ok: bool
    n_solids: int
    volume: float
    occt_valid: bool
    watertight: bool
    reason: str = ""


def check_validity(shape, mesh=None) -> Validity:
    try:
        sols = solids_of(shape)
        n = len(sols)
        vol = float(shape.Volume())
        occt_valid = bool(shape.isValid())
    except Exception as e:  # noqa: BLE001
        return Validity(False, 0, 0.0, False, False, f"occt error {e}")
    if mesh is None:
        try:
            mesh = shape_to_mesh(shape, 0.05)
        except Exception:  # noqa: BLE001
            mesh = None
    wt = bool(mesh is not None and mesh.is_watertight)
    reason = ""
    if n != 1:
        reason = f"{n} solids (need exactly 1)"
    elif vol <= 1e-3:
        reason = "zero/negative volume"
    elif not occt_valid:
        reason = "OCCT BRepCheck invalid"
    elif not wt:
        reason = "tessellation not watertight"
    return Validity(reason == "", n, vol, occt_valid, wt, reason)


class OCCTQuery:
    """Point classification + ray casting against a B-rep solid (primary verifier backend)."""

    def __init__(self, shape, tol: float = 1e-4):
        from OCP.BRepClass3d import BRepClass3d_SolidClassifier
        from OCP.IntCurvesFace import IntCurvesFace_ShapeIntersector

        self.shape = shape
        self.wrapped = shape.wrapped
        self.tol = tol
        self._cls = BRepClass3d_SolidClassifier(self.wrapped)
        self._isect = IntCurvesFace_ShapeIntersector()
        self._isect.Load(self.wrapped, tol)

    def inside(self, p) -> bool:
        from OCP.gp import gp_Pnt
        from OCP.TopAbs import TopAbs_IN, TopAbs_ON

        self._cls.Perform(gp_Pnt(float(p[0]), float(p[1]), float(p[2])), self.tol)
        st = self._cls.State()
        return st == TopAbs_IN or st == TopAbs_ON

    def ray_first_hit(self, p, d, tmax: float = 1e3) -> float:
        """Distance along unit direction d from p to the first surface hit (inf if none)."""
        from OCP.gp import gp_Dir, gp_Lin, gp_Pnt

        lin = gp_Lin(gp_Pnt(*map(float, p)), gp_Dir(*map(float, d)))
        self._isect.Perform(lin, 0.0, tmax)
        if not self._isect.IsDone() or self._isect.NbPnt() == 0:
            return float("inf")
        return min(self._isect.WParameter(i) for i in range(1, self._isect.NbPnt() + 1))


class MeshQuery:
    """Same interface as OCCTQuery but on a triangle mesh (held-out backend)."""

    def __init__(self, mesh):
        self.mesh = mesh
        try:
            from trimesh.ray.ray_triangle import RayMeshIntersector

            self._ray = RayMeshIntersector(mesh)
        except Exception:  # noqa: BLE001
            self._ray = mesh.ray

    def inside_many(self, pts: np.ndarray) -> np.ndarray:
        return self.mesh.contains(np.asarray(pts, float))

    def ray_first_hit_many(self, pts: np.ndarray, dirs: np.ndarray, tmax: float = 1e3) -> np.ndarray:
        pts = np.asarray(pts, float)
        dirs = np.asarray(dirs, float)
        out = np.full(len(pts), np.inf)
        locs, idx_ray, _ = self._ray.intersects_location(pts, dirs, multiple_hits=True)
        if len(idx_ray):
            t = np.einsum("ij,ij->i", locs - pts[idx_ray], dirs[idx_ray])
            for r, tt in zip(idx_ray, t):
                if 1e-9 < tt < out[r] and tt <= tmax:
                    out[r] = tt
        return out


def signed_directional_gap_occt(q: OCCTQuery, pts: np.ndarray, normals: np.ndarray, search: float = 5.0) -> np.ndarray:
    """For each mating-surface point p with normal n (pointing *into the gap*):
    gap = +distance along n to the part if p is outside part material,
          -distance along -n to the part boundary if p is inside part material (interference).
    Returns +inf where no part surface lies within ``search`` mm (feature missing)."""
    gaps = np.empty(len(pts))
    for i, (p, n) in enumerate(zip(pts, normals)):
        if q.inside(p):
            t = q.ray_first_hit(p, -n, search)
            gaps[i] = -t if np.isfinite(t) else -search
        else:
            t = q.ray_first_hit(p, n, search)
            gaps[i] = t if t <= search else np.inf
    return gaps


def signed_directional_gap_mesh(mq: MeshQuery, pts: np.ndarray, normals: np.ndarray, search: float = 5.0) -> np.ndarray:
    pts = np.asarray(pts, float)
    normals = np.asarray(normals, float)
    inside = mq.inside_many(pts)
    fwd = mq.ray_first_hit_many(pts, normals, search)
    bwd = mq.ray_first_hit_many(pts, -normals, search)
    gaps = np.where(inside, -np.minimum(bwd, search), np.where(fwd <= search, fwd, np.inf))
    return gaps


def bbox(shape) -> tuple[np.ndarray, np.ndarray]:
    bb = shape.BoundingBox()
    return np.array([bb.xmin, bb.ymin, bb.zmin]), np.array([bb.xmax, bb.ymax, bb.zmax])
