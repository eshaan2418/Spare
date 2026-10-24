"""Composite verifier suite (primary = reward; heldout = evaluation only).

A *task* is everything needed to judge a generated program without looking at
its parameters: fit specs on mating geometry, geometric probes, strength checks
and load case, the printer profile and material.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from spare.common.geom import check_validity, shape_to_mesh
from spare.common.sandbox import ExecResult, run_program_local
from spare.features.tolerance import DEFAULT_PRINTER, VIRTUAL_PRINTERS, PrinterProfile
from spare.synth.family import PartInstance
from spare.synth.registry import get as get_family
from spare.verifiers.editability.editability import functional_editability, static_editability
from spare.verifiers.fit.fit import check_fit
from spare.verifiers.geometry.probes import MeshProber, OCCTProber, dims_score, run_probes
from spare.verifiers.printability.printability import analyse_geometric, analyse_slicer
from spare.verifiers.strength.strength import check_strength_analytic, check_strength_fea

WEIGHTS = {"fit": 0.35, "dims": 0.20, "print": 0.15, "strength": 0.15, "edit": 0.15}


@dataclass
class Task:
    inst: PartInstance
    printer: PrinterProfile = DEFAULT_PRINTER
    material: str = "PLA"

    @property
    def family(self):
        return get_family(self.inst.family)

    def fit_specs(self):
        return self.family.fit_specs(self.inst.mating, self.inst.design)

    def probes(self):
        return self.family.probes(self.inst.design)

    def strength_checks(self):
        return self.family.strength_checks(self.inst.mating, self.inst.design)

    def load_case(self):
        return self.family.load_case(self.inst.mating, self.inst.design)

    _mating = None

    def mating_shape(self):
        if self._mating is None:
            r = run_program_local(self.family.mating_program(self.inst.mating, self.inst.design), trusted=True)
            self._mating = r.shape() if r.ok else None
        return self._mating


@dataclass
class Report:
    ok_exec: bool
    error: str = ""
    error_kind: str = ""
    valid: bool = False
    validity_reason: str = ""
    fit_pass: bool = False
    fit_score: float = 0.0
    fit_detail: list = field(default_factory=list)
    dims_score: float = 0.0
    dims_pass: bool = False
    dims_detail: list = field(default_factory=list)
    print_pass: bool = False
    print_score: float = 0.0
    print_detail: dict = field(default_factory=dict)
    strength_pass: bool = False
    strength_score: float = 0.0
    strength_util: float = float("nan")
    edit_score: float = 0.0
    edit_pass: bool = False
    edit_detail: dict = field(default_factory=dict)
    reward: float = 0.0
    all_pass: bool = False
    suite: str = "primary"
    seconds: float = 0.0

    def to_dict(self):
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, float) and not math.isfinite(v):
                d[k] = None
        return d


def compose_reward(r: Report, weights: dict | None = None) -> float:
    w = weights or WEIGHTS
    if not r.ok_exec:
        return 0.0
    if not r.valid:
        return 0.05  # executes but not a single valid solid
    tot = sum(w.values())
    s = (w.get("fit", 0) * r.fit_score + w.get("dims", 0) * r.dims_score + w.get("print", 0) * r.print_score
         + w.get("strength", 0) * r.strength_score + w.get("edit", 0) * r.edit_score)
    return float(0.1 + 0.9 * s / max(tot, 1e-9))


def verify(src: str, task: Task, exec_result: ExecResult | None = None, suite: str = "primary",
           weights: dict | None = None, printer: PrinterProfile | None = None, slicer: bool = True,
           fea: bool = True, functional_edit: bool = True, seed: int = 0) -> Report:
    """Run the verifier suite on a program. ``exec_result`` lets callers pass a pooled execution."""
    t0 = time.time()
    er = exec_result or run_program_local(src, trusted=False)
    rep = Report(ok_exec=er.ok, error=er.error, error_kind=er.error_kind, suite=suite)
    if not er.ok:
        rep.seconds = time.time() - t0
        return rep
    shape = er.shape()
    pr = printer or task.printer
    heldout = suite == "heldout"
    mesh = shape_to_mesh(shape, tol=0.01 if heldout else 0.05, ang_tol=0.1 if heldout else 0.2)
    v = check_validity(shape, mesh)
    rep.valid, rep.validity_reason = v.ok, v.reason
    if not v.ok:
        rep.reward = compose_reward(rep, weights)
        rep.seconds = time.time() - t0
        return rep
    specs = task.fit_specs()
    fit = check_fit(shape, specs, task.mating_shape(), pr, backend="mesh" if heldout else "occt", mesh=mesh, seed=seed)
    rep.fit_pass, rep.fit_score = fit.passed, fit.score
    rep.fit_detail = [asdict(s) for s in fit.specs] + [{"interference": fit.interference_volume,
                                                       "allowed": fit.interference_allowed, "ok": fit.interference_ok}]
    prober = MeshProber(mesh) if heldout else OCCTProber(shape)
    pres = run_probes(prober, task.probes())
    rep.dims_score = dims_score(pres)
    rep.dims_pass = all(p.ok for p in pres if p.critical)
    rep.dims_detail = [asdict(p) for p in pres]
    if heldout and slicer:
        p = analyse_slicer(mesh, "generic_petg_0.4" if task.material == "PETG" else "generic_pla_0.4")
    else:
        p = analyse_geometric(mesh, shape=shape, seed=seed)
    rep.print_pass, rep.print_score, rep.print_detail = p.passed, p.score, {**p.metrics, "reasons": p.reasons,
                                                                           "backend": p.backend}
    if heldout and fea:
        st = check_strength_fea(shape, task.load_case(), task.material, order=1)
        if not task.load_case():  # families without a load case: fall back to analytic (documented)
            st = check_strength_analytic(OCCTProber(shape), task.strength_checks(), task.material, fit)
    else:
        st = check_strength_analytic(OCCTProber(shape), task.strength_checks(), task.material, fit)
    rep.strength_pass, rep.strength_score = st.passed, st.score
    rep.strength_util = st.utilisation
    if heldout and functional_edit:
        def run_fn(s):
            r = run_program_local(s, trusted=False)
            if not r.ok:
                return None
            sh = r.shape()
            return sh if check_validity(sh).ok else None

        def gap_fn(sh):
            f = check_fit(sh, specs, None, pr, backend="occt")
            return [s.median_printed_gap for s in f.specs]

        e = functional_editability(src, gap_fn, run_fn)
    else:
        e = static_editability(src)
    rep.edit_score, rep.edit_pass, rep.edit_detail = e.score, e.passed, e.metrics
    rep.all_pass = bool(rep.valid and rep.fit_pass and rep.print_pass and rep.strength_pass)
    rep.reward = compose_reward(rep, weights)
    rep.seconds = time.time() - t0
    return rep


def virtual_printer_for(uid: str) -> PrinterProfile:
    """Deterministic assignment of a held-out virtual printer to a test part."""
    keys = sorted(VIRTUAL_PRINTERS)
    h = sum(ord(c) * (i + 1) for i, c in enumerate(uid))
    return VIRTUAL_PRINTERS[keys[h % len(keys)]]


_ = np
