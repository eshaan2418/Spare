"""Strength verifiers.

Primary (used as RL reward): analytic formulas whose geometric inputs are
*measured on the generated solid* with ray probes.
Held-out (never used as reward): linear-elastic FEA with gmsh tetrahedra and
scikit-fem, on the family's load case.

Utilisation u = demand / allowable; pass iff u <= 1.
"""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from spare.features.tolerance import MATERIALS
from spare.synth.family import LoadCase, StrengthCheck
from spare.verifiers.geometry.probes import measure

SAFETY_FACTOR = 2.0


@dataclass
class StrengthResult:
    passed: bool
    utilisation: float
    checks: list[dict] = field(default_factory=list)
    backend: str = "analytic"
    note: str = ""

    @property
    def score(self) -> float:
        u = self.utilisation
        if not math.isfinite(u):
            return 0.0
        return 1.0 if u <= 1.0 else float(math.exp(-2.0 * (u - 1.0)))


def _eval_check(prober, sc: StrengthCheck, material: str, fit=None) -> dict:
    mat = MATERIALS[material]
    m = {k: measure(prober, p) for k, p in sc.probes.items()}
    c = sc.consts
    out: dict[str, Any] = {"name": sc.name, "formula": sc.formula, "measured": m}
    try:
        if sc.formula == "cantilever_strain":
            t = m["t"]
            if "tip" in m:  # battery cover: ray from x_w + off toward -x hits the lip tip
                y = c["x_w_off"] - m["tip"]
            else:
                y = (c["deflect_target"] - 2 * m["half_aperture"]) / 2
            y = max(y, 0.0)
            eps = 1.5 * t * y / c["L"] ** 2
            out.update(demand=eps, allowable=mat.allow_strain_snap, deflection=y)
        elif sc.formula == "cantilever_stress":
            sig = 6 * c["F"] * c["L"] / (m["b"] * m["t"] ** 2)
            out.update(demand=sig, allowable=mat.yield_mpa / SAFETY_FACTOR)
        elif sc.formula == "hoop_strain":
            g = fit.specs[c["fit_spec"]].median_printed_gap if fit is not None and fit.specs else 0.0
            eps = max(-g, 0.0) / c["radius"]
            out.update(demand=eps, allowable=mat.allow_strain_snap)
        elif sc.formula == "wall_flex":
            g = fit.specs[c["fit_spec"]].median_printed_gap if fit is not None and fit.specs else 0.0
            # wall as a fixed-fixed beam deflected by the interference at mid-span: eps = 12 t delta / L^2
            eps = 12.0 * m["t"] * max(-g, 0.0) / c["span"] ** 2
            out.update(demand=eps, allowable=mat.allow_strain_snap)
        else:
            raise ValueError(sc.formula)
    except (KeyError, ZeroDivisionError, TypeError):
        out.update(demand=float("inf"), allowable=1.0)
    d = out["demand"]
    out["utilisation"] = float(d / out["allowable"]) if math.isfinite(d) else float("inf")
    return out


def check_strength_analytic(prober, checks: list[StrengthCheck], material: str = "PLA", fit=None) -> StrengthResult:
    if not checks:
        return StrengthResult(True, 0.0, [], "analytic", "no load case")
    res = [_eval_check(prober, sc, material, fit) for sc in checks]
    u = max(r["utilisation"] for r in res)
    return StrengthResult(bool(u <= 1.0), float(u), res, "analytic")


# ------------------------------------------------------------------------ FEA (held-out)
def _tet_mesh(shape, h: float):
    import gmsh

    with tempfile.TemporaryDirectory() as td:
        fn = os.path.join(td, "part.step")
        shape.exportStep(fn)
        gmsh.initialize(interruptible=False)
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.open(fn)
            gmsh.option.setNumber("Mesh.MeshSizeMax", h)
            gmsh.option.setNumber("Mesh.MeshSizeMin", h * 0.3)
            gmsh.option.setNumber("Mesh.Algorithm3D", 1)
            gmsh.model.mesh.generate(3)
            ntags, coords, _ = gmsh.model.mesh.getNodes()
            et, _, enodes = gmsh.model.mesh.getElements(3)
            tets = None
            for t, en in zip(et, enodes):
                if t == 4:
                    tets = np.asarray(en, np.int64).reshape(-1, 4)
            if tets is None:
                raise RuntimeError("no tetrahedra")
            idx = {int(t): i for i, t in enumerate(ntags)}
            p = np.asarray(coords).reshape(-1, 3)
            t = np.vectorize(idx.get)(tets)
            used = np.unique(t)
            remap = -np.ones(len(p), np.int64)
            remap[used] = np.arange(len(used))
            return p[used], remap[t]
        finally:
            gmsh.finalize()


def _in_box(p: np.ndarray, b: list[float]) -> np.ndarray:
    return (p[:, 0] >= b[0]) & (p[:, 1] >= b[1]) & (p[:, 2] >= b[2]) & (p[:, 0] <= b[3]) & (p[:, 1] <= b[4]) & (p[:, 2] <= b[5])


def check_strength_fea(shape, lc: LoadCase | None, material: str = "PLA", h: float | None = None,
                       order: int = 1) -> StrengthResult:
    if lc is None:
        return StrengthResult(True, 0.0, [], "fea", "no load case")
    from skfem import Basis, ElementTetP1, ElementTetP2, ElementVector, MeshTet, asm, condense, solve
    from skfem.helpers import ddot, sym_grad, trace
    from skfem.models.elasticity import lame_parameters, linear_elasticity

    mat = MATERIALS[material]
    if lc.crop_box is not None:
        import cadquery as cq

        b = lc.crop_box
        lo = [max(b[i], -500) for i in range(3)]
        hi = [min(b[i + 3], 500) for i in range(3)]
        box = cq.Solid.makeBox(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], cq.Vector(*lo))
        shape = shape.intersect(box)
    bb = shape.BoundingBox()
    h = h or lc.h or max(min(bb.xlen, bb.ylen, bb.zlen) / 4.0, 0.5)
    try:
        p, t = _tet_mesh(shape, h)
    except Exception as e:  # noqa: BLE001
        return StrengthResult(False, float("inf"), [], "fea", f"meshing failed: {e}")
    mesh = MeshTet(p.T.copy(), t.T.copy())
    el = ElementVector(ElementTetP2() if order == 2 else ElementTetP1())
    basis = Basis(mesh, el)
    lam, mu = lame_parameters(mat.E_mpa, 0.35)
    K = asm(linear_elasticity(lam, mu), basis)
    dofs_xyz = basis.nodal_dofs  # (3, nnodes) for vertex nodes
    P = mesh.p.T
    fixed_nodes = np.where(_in_box(P, lc.fixed_box))[0]
    load_nodes = np.where(_in_box(P, lc.load_box))[0]
    if len(fixed_nodes) < 3 or len(load_nodes) < 1:
        return StrengthResult(False, float("inf"), [], "fea", "load/fixed region not on part")
    D: Any = list(dofs_xyz[:, fixed_nodes].ravel())
    if order == 2:  # also fix edge dofs whose midpoints lie in the fixed box
        mids = 0.5 * (P[mesh.edges[0]] + P[mesh.edges[1]])
        fe = np.where(_in_box(mids, lc.fixed_box))[0]
        D += list(basis.edge_dofs[:, fe].ravel())
    D = np.unique(np.array(D, np.int64))
    u = np.zeros(basis.N)
    f = np.zeros(basis.N)
    vec = np.asarray(lc.vector, float)
    if lc.mode == "force":
        share = vec / len(load_nodes)
        for k in range(3):
            f[dofs_xyz[k, load_nodes]] += share[k]
        x = solve(*condense(K, f, D=D))
    else:
        comp = int(np.argmax(np.abs(vec)))
        pd = dofs_xyz[comp, load_nodes]
        u[pd] = vec[comp]
        D2 = np.unique(np.r_[D, pd])
        x = solve(*condense(K, f, x=u, D=D2))
    # element strains / stresses at quadrature points
    from skfem import ElementTetP0

    b0 = basis.with_element(ElementTetP0())
    ui = basis.interpolate(x)
    E = sym_grad(ui)
    S = 2 * mu * E + lam * trace(E) * np.eye(3)[:, :, None, None] if False else None  # noqa: F841
    tr = E[0, 0] + E[1, 1] + E[2, 2]
    sxx = 2 * mu * E[0, 0] + lam * tr
    syy = 2 * mu * E[1, 1] + lam * tr
    szz = 2 * mu * E[2, 2] + lam * tr
    sxy, syz, sxz = 2 * mu * E[0, 1], 2 * mu * E[1, 2], 2 * mu * E[0, 2]
    vm = np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3 * (sxy ** 2 + syz ** 2 + sxz ** 2))
    eps_eq = np.sqrt(2.0 / 3.0 * ddot(E, E))
    _ = b0
    vm_p = float(np.percentile(vm.max(axis=1), 99))
    ep_p = float(np.percentile(eps_eq.max(axis=1), 99))
    if lc.criterion == "stress":
        dem, allow = vm_p, mat.yield_mpa / SAFETY_FACTOR
    else:
        dem, allow = ep_p, mat.allow_strain_snap
    util = dem / allow
    return StrengthResult(bool(util <= 1.0), float(util),
                          [{"name": lc.name, "vm_p99_mpa": vm_p, "strain_p99": ep_p, "demand": dem, "allowable": allow,
                            "nodes": int(mesh.p.shape[1]), "h": h, "order": order}], "fea")
