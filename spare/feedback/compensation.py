"""Per-printer/material compensation learned from fit outcomes.

Each print test of a fit feature gives an interval observation of the printed per-side gap:
  fits -> gap in [lo, hi];  too_tight -> gap < lo;  too_loose -> gap > hi,
with printed gap = designed gap - (model deviation(kind, size) + b_kind). We estimate the per-surface-kind
bias b_kind by maximising a censored-Gaussian likelihood with a zero-mean prior (MAP, grid search).
The bias feeds the tolerance engine (``design_allowance(..., bias)``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from spare.features.tolerance import DEFAULT_PRINTER, FIT_BANDS, PrinterProfile, printed_deviation

GRID = np.round(np.arange(-0.4, 0.4001, 0.005), 3)


def _phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


@dataclass
class Observation:
    kind: str
    size: float
    fit_class: str
    designed_gap: float
    outcome: str  # fits | too_tight | too_loose


@dataclass
class CompensationModel:
    base: PrinterProfile = DEFAULT_PRINTER
    noise: float = 0.05  # print-to-print per-side variation (mm), assumption
    prior_sd: float = 0.15
    obs: list[Observation] = field(default_factory=list)
    bias: dict[str, float] = field(default_factory=dict)

    def add(self, o: Observation):
        self.obs.append(o)

    def fit(self) -> dict[str, float]:
        for kind in sorted({o.kind for o in self.obs}):
            ob = [o for o in self.obs if o.kind == kind]
            best, bl = 0.0, -1e18
            for b in GRID:
                ll = -0.5 * (b / self.prior_sd) ** 2
                for o in ob:
                    lo, hi = FIT_BANDS[o.fit_class]
                    g = o.designed_gap - printed_deviation(o.kind, o.size, self.base) - b
                    if o.outcome == "fits":
                        p = _phi((hi - g) / self.noise) - _phi((lo - g) / self.noise)
                    elif o.outcome == "too_tight":
                        p = _phi((lo - g) / self.noise)
                    else:
                        p = 1 - _phi((hi - g) / self.noise)
                    ll += math.log(max(p, 1e-12))
                if ll > bl:
                    best, bl = float(b), ll
            self.bias[kind] = best
        return dict(self.bias)

    def deviation(self, kind: str, size: float) -> float:
        return printed_deviation(kind, size, self.base) + self.bias.get(kind, 0.0)


def true_bias(kind: str, size: float, truth: PrinterProfile, base: PrinterProfile = DEFAULT_PRINTER) -> float:
    return printed_deviation(kind, size, truth) - printed_deviation(kind, size, base)
