"""Procedural part families: the contract every family implements.

A family turns a parameter dict into
  * a CadQuery *program* (named parameters, canonical print frame: Z up = build
    direction, part on z >= 0),
  * mating geometry (shaft, tube, housing, ...) as a CadQuery program,
  * fit specifications: samples on the mating *functional surface* with normals
    pointing into the gap, the fit class and surface kind,
  * probes: critical/cosmetic dimensions measured geometrically on any solid,
  * strength checks (analytic, measured from the solid) and an FEA load case,
  * a family-agnostic feature graph (the structured representation fed to the
    generator).

Parameters come in three layers:
  ``mating``   -- the counterpart dims (what the part must fit)
  ``original`` -- the part as manufactured (what photos show; molded tolerances)
  ``design``   -- the dims to print, derived by the tolerance engine from mating
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from spare.features.tolerance import DEFAULT_PRINTER, MATERIALS, PrinterProfile, design_allowance


@dataclass
class FitSpec:
    name: str
    fit_class: str
    kind: str
    points: np.ndarray
    normals: np.ndarray
    feature_size: float
    param: str  # design parameter that controls this gap (for feedback / editing)
    location: str  # human label used in fit reports ("shaft", "cable", ...)
    gap_sign: int = 1  # d(gap)/d(param): +1 if increasing param increases the gap (per side, x0.5 for diameters)
    param_scale: float = 0.5  # per-side gap change per unit param change


@dataclass
class Probe:
    name: str
    kind: str  # extent | circle_d | circle_cx | circle_cy | thickness
    args: dict[str, Any]
    target: float
    tol: float
    critical: bool = True


@dataclass
class LoadCase:
    """FEA load case in the canonical frame (boxes are [xmin,ymin,zmin,xmax,ymax,zmax])."""
    name: str
    fixed_box: list[float]
    load_box: list[float]
    mode: str  # "force" (N, total) or "disp" (mm prescribed)
    vector: list[float]
    criterion: str  # "stress" or "strain"
    crop_box: list[float] | None = None  # analyse only the part inside this box (fixed region must lie in it)
    h: float | None = None  # target element size (mm)


@dataclass
class StrengthCheck:
    name: str
    formula: str  # cantilever_strain | cantilever_stress | hoop_strain
    probes: dict[str, Probe]  # geometry measured from the solid feeding the formula
    consts: dict[str, float]


@dataclass
class PartInstance:
    family: str
    uid: str
    split: str
    mating: dict[str, float]
    original: dict[str, float]
    design: dict[str, float]
    style: int = 0
    material: str = "PLA"
    meta: dict[str, Any] = field(default_factory=dict)


def r2(x: float) -> float:
    return float(round(x, 2))


def cyl_points(center, axis: str, r: float, z0: float, z1: float, n_ang: int = 24, n_ax: int = 5,
               ang_range: tuple[float, float] = (0, 2 * math.pi), outward: bool = True,
               keep=None) -> tuple[np.ndarray, np.ndarray]:
    """Points on a cylinder surface (axis 'x'|'y'|'z' through ``center``) with radial normals.
    outward=True -> normals point away from the axis (shaft in a bore);
    outward=False -> toward the axis (part plug inside a mating hole)."""
    pts, nrm = [], []
    a0, a1 = ang_range
    full = abs((a1 - a0) - 2 * math.pi) < 1e-9
    angs = np.linspace(a0, a1, n_ang, endpoint=not full)
    for t in np.linspace(z0, z1, n_ax):
        for a in angs:
            u, v = math.cos(a), math.sin(a)
            if axis == "z":
                d = np.array([u, v, 0.0])
                p = np.array([center[0] + r * u, center[1] + r * v, t])
            elif axis == "y":
                d = np.array([u, 0.0, v])
                p = np.array([center[0] + r * u, t, center[2] + r * v])
            else:
                d = np.array([0.0, u, v])
                p = np.array([t, center[1] + r * u, center[2] + r * v])
            if keep is not None and not keep(p):
                continue
            pts.append(p)
            nrm.append(d if outward else -d)
    return np.array(pts), np.array(nrm)


def plane_points(origin, u, v, nu: int, nv: int, normal) -> tuple[np.ndarray, np.ndarray]:
    """Grid on a planar patch origin + s*u + t*v, s,t in [0,1]."""
    o, u, v = np.asarray(origin, float), np.asarray(u, float), np.asarray(v, float)
    pts = [o + s * u + t * v for s in np.linspace(0, 1, nu) for t in np.linspace(0, 1, nv)]
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    return np.array(pts), np.tile(n, (len(pts), 1))


class Family:
    name: str = ""
    split: str = "train"
    part_type: str = ""  # human-readable type hint
    n_styles: int = 1

    # ---- parameter generation -------------------------------------------------
    def sample_mating(self, rng: np.random.Generator) -> dict[str, float]:
        raise NotImplementedError

    def sample_original(self, rng: np.random.Generator, m: dict[str, float]) -> dict[str, float]:
        raise NotImplementedError

    def derive_design(self, m: dict[str, float], o: dict[str, float], pr: PrinterProfile,
                      material: str = "PLA", bias: dict[str, float] | None = None) -> dict[str, float]:
        """Apply the tolerance engine: original cosmetic dims + mating-derived critical dims."""
        raise NotImplementedError

    def instance(self, rng: np.random.Generator, uid: str, pr: PrinterProfile = DEFAULT_PRINTER,
                 material: str = "PLA") -> PartInstance:
        m = self.sample_mating(rng)
        o = self.sample_original(rng, m)
        d = self.derive_design(m, o, pr, material)
        return PartInstance(self.name, uid, self.split, m, o, d, int(rng.integers(self.n_styles)), material)

    # ---- geometry --------------------------------------------------------------
    def program(self, p: dict[str, float], style: int = 0) -> str:
        raise NotImplementedError

    def mating_program(self, m: dict[str, float], d: dict[str, float]) -> str:
        raise NotImplementedError

    def fit_specs(self, m: dict[str, float], d: dict[str, float]) -> list[FitSpec]:
        raise NotImplementedError

    def probes(self, d: dict[str, float]) -> list[Probe]:
        raise NotImplementedError

    def strength_checks(self, m: dict[str, float], d: dict[str, float]) -> list[StrengthCheck]:
        return []

    def load_case(self, m: dict[str, float], d: dict[str, float]) -> LoadCase | None:
        return None

    def feature_graph(self, m: dict[str, float], d: dict[str, float], o: dict[str, float] | None = None) -> dict:
        raise NotImplementedError

    def break_program(self, o: dict[str, float], rng: np.random.Generator) -> tuple[str, dict]:
        """Program for a *broken* original (for renders / completion tests)."""
        return self.program(o), {"type": "none"}

    # ---- helpers ------------------------------------------------------------------
    @staticmethod
    def allow(fit_class: str, kind: str, size: float, pr: PrinterProfile, bias: dict | None, key: str) -> float:
        b = (bias or {}).get(key, 0.0)
        return design_allowance(fit_class, kind, size, pr, b)

    @staticmethod
    def material(name: str):
        return MATERIALS[name]


def header(title: str, frame: str) -> str:
    return f"import cadquery as cq\n\n# {title}\n# Units: mm. Frame: {frame}\n"


def fmt_params(params: list[tuple[str, float, str]]) -> str:
    lines = []
    for name, val, comment in params:
        v = f"{val:.2f}".rstrip("0").rstrip(".") if isinstance(val, float) else str(val)
        if "." not in v and isinstance(val, float):
            v += ".0"
        lines.append(f"{name} = {v}" + (f"  # {comment}" if comment else ""))
    return "\n".join(lines) + "\n"
