"""Tolerance / clearance engine and FDM printer deviation model.

Conventions
-----------
* A *signed gap* is measured per side (radially for round features), in mm,
  between a mating surface and the part surface. Positive = air gap,
  negative = interference.
* ``FIT_BANDS[fit_class]`` is the acceptable *printed* signed gap range for the
  fit to function (e.g. a knob on a D-shaft must be snug but removable).
* The printer deviates from the design: material grows into the gap by
  ``printed_deviation(kind, size, printer)`` mm per side. Holes in FDM print
  undersize (a well-documented effect); the model is a first-order one with
  per-printer parameters that the feedback loop (M7) learns.

The band and default values are engineering defaults drawn from common FDM
practice (see docs/PRIOR_WORK.md, DFAM section). They are *assumptions*, logged
in DECISIONS.md, and the compensation part is learned per printer in M7.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

# acceptable printed per-side signed gap (mm)
FIT_BANDS: dict[str, tuple[float, float]] = {
    "clearance": (0.10, 0.45),   # free rotation / sliding on pin, bolt, axle
    "loose": (0.10, 0.70),       # screw holes, coarse alignment
    "transition": (-0.05, 0.15),  # snug, removable (knob on shaft, clip on cable)
    "press": (-0.25, -0.03),     # permanent friction fit (caps, plugs)
    "slide": (0.08, 0.60),       # covers sliding into openings
    "seat": (0.00, 0.50),        # snap lip seating against a catch face (axial play)
}

# what kind of surface carries the fit -> determines printer deviation
SURFACE_KINDS = ("hole", "hole_h", "outer", "outer_h", "z")


@dataclass(frozen=True)
class PrinterProfile:
    name: str = "generic_pla_0.4"
    material: str = "PLA"
    nozzle: float = 0.4
    e_xy: float = 0.07         # outward growth of vertical walls per side (mm)
    hole_c: float = 0.03       # extra radial shrink of vertical holes (mm)
    hole_k: float = 0.25       # extra radial shrink ~ hole_k / diameter (mm^2)
    e_z: float = 0.05          # growth of horizontal faces into gaps (mm)
    hole_h_extra: float = 0.08  # extra shrink of horizontal holes (layer stepping / sag)

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_PRINTER = PrinterProfile()

# Fixed virtual printers used for *simulated* evaluation; the generator never
# sees which one a test part is assigned to (mirrors not knowing a user's printer).
VIRTUAL_PRINTERS: dict[str, PrinterProfile] = {
    "vp_tight": PrinterProfile("vp_tight", "PLA", 0.4, 0.13, 0.06, 0.35, 0.08, 0.13),
    "vp_nominal": PrinterProfile("vp_nominal", "PLA", 0.4, 0.07, 0.03, 0.25, 0.05, 0.08),
    "vp_loose": PrinterProfile("vp_loose", "PETG", 0.4, 0.02, 0.01, 0.12, 0.02, 0.04),
}


@dataclass(frozen=True)
class Material:
    name: str
    E_mpa: float
    yield_mpa: float
    z_strength_factor: float  # inter-layer strength / in-layer strength
    allow_strain_snap: float  # one-time-ish snap assembly strain
    max_temp_c: float


MATERIALS = {
    "PLA": Material("PLA", 3500, 50, 0.55, 0.015, 55),
    "PETG": Material("PETG", 2100, 45, 0.65, 0.030, 70),
    "ABS": Material("ABS", 2200, 40, 0.60, 0.025, 85),
}


def printed_deviation(kind: str, size: float, pr: PrinterProfile = DEFAULT_PRINTER) -> float:
    """Per-side growth of part material into the gap (mm) for a surface kind.
    ``size`` is the feature diameter/width (mm)."""
    size = max(size, 0.5)
    if kind == "hole":
        return pr.e_xy + pr.hole_c + pr.hole_k / size
    if kind == "hole_h":
        return 0.5 * (pr.e_xy + pr.e_z) + pr.hole_h_extra + pr.hole_k / size
    if kind == "outer":
        return pr.e_xy
    if kind == "outer_h":
        return 0.5 * (pr.e_xy + pr.e_z)
    if kind == "z":
        return pr.e_z
    raise ValueError(kind)


def band_center(fit_class: str) -> float:
    lo, hi = FIT_BANDS[fit_class]
    return 0.5 * (lo + hi)


def design_allowance(fit_class: str, kind: str, size: float, pr: PrinterProfile = DEFAULT_PRINTER,
                     bias: float = 0.0) -> float:
    """Designed per-side gap so that the *printed* gap lands mid-band.
    ``bias`` is a learned per-printer correction (M7) added on top."""
    return band_center(fit_class) + printed_deviation(kind, size, pr) + bias


def with_bias(pr: PrinterProfile, **kw) -> PrinterProfile:
    return replace(pr, **kw)


def press_allowance(kind: str, size: float, radius: float, allow_strain: float,
                    pr: PrinterProfile = DEFAULT_PRINTER, bias: float = 0.0, safety: float = 0.8) -> float:
    """Designed per-side gap for a press fit, limiting interference so that the hoop/wall strain
    stays under ``safety * allow_strain`` (FDM parts split easily). Never looser than the band's hi."""
    lo, hi = FIT_BANDS["press"]
    target = band_center("press")
    target = max(target, -safety * allow_strain * radius)
    target = min(target, hi - 0.02)
    return target + printed_deviation(kind, size, pr) + bias
