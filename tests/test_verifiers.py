"""Verifier unit tests on known-good and adversarially bad parts."""
import numpy as np
import pytest

from spare.features.tolerance import (
    DEFAULT_PRINTER,
    FIT_BANDS,
    band_center,
    design_allowance,
    printed_deviation,
)
from spare.synth.registry import FAMILIES
from spare.verifiers.editability.editability import static_editability
from spare.verifiers.suite import Task, verify


def _task(instances, name):
    return Task(instances[name])


def test_tolerance_allowance_lands_mid_band():
    for fc in FIT_BANDS:
        a = design_allowance(fc, "hole", 6.0)
        assert abs((a - printed_deviation("hole", 6.0, DEFAULT_PRINTER)) - band_center(fc)) < 1e-9


def test_good_part_passes_primary(instances):
    t = _task(instances, "spacer")
    r = verify(FAMILIES["spacer"].program(t.inst.design), t)
    assert r.all_pass and r.reward > 0.95


def test_bore_too_small_is_too_tight(instances):
    t = _task(instances, "spacer")
    d = dict(t.inst.design, bore_diameter=t.inst.mating["bolt_diameter"] - 0.1)
    r = verify(FAMILIES["spacer"].program(d), t)
    assert r.valid and not r.fit_pass
    assert r.fit_detail[0]["verdict"] == "too_tight"


def test_bore_too_big_is_too_loose(instances):
    t = _task(instances, "spacer")
    d = dict(t.inst.design, bore_diameter=t.inst.mating["bolt_diameter"] + 2.0)
    r = verify(FAMILIES["spacer"].program(d), t)
    assert not r.fit_pass and r.fit_detail[0]["verdict"] in ("too_loose", "missing")


def test_featureless_block_fails_fit(instances):
    t = _task(instances, "knob_dshaft")
    d = t.inst.design
    src = f"import cadquery as cq\nresult = cq.Workplane('XY').circle({d['knob_diameter'] / 2}).extrude({d['knob_height']})\n"
    r = verify(src, t)
    assert r.valid and not r.fit_pass
    assert not r.fit_detail[-1]["ok"]  # massive interference with the shaft


def test_disjoint_solids_invalid(instances):
    t = _task(instances, "spacer")
    src = "import cadquery as cq\nresult = cq.Workplane().box(5,5,5).union(cq.Workplane().box(5,5,5).translate((20,0,0)))\n"
    r = verify(src, t)
    assert r.ok_exec and not r.valid and r.reward <= 0.05


def test_exec_error_zero_reward(instances):
    r = verify("import cadquery as cq\nresult = cq.Workplane().box(1,1,1).fillet(3)\n", _task(instances, "spacer"))
    assert not r.ok_exec and r.reward == 0.0


def test_thin_wall_fails_printability(instances):
    t = _task(instances, "spacer")
    b = t.inst.mating["bolt_diameter"]
    src = (f"import cadquery as cq\nbore = {b + 0.55}\nwall = 0.3\nheight = {t.inst.design['height']}\n"
           "result = cq.Workplane('XY').circle(bore / 2 + wall).circle(bore / 2).extrude(height)\n")
    r = verify(src, t)
    assert r.valid and not r.print_pass


def test_overhang_fails_printability(instances):
    t = _task(instances, "spacer")
    src = ("import cadquery as cq\nstem = 4.0\ncap = 40.0\n"
           "result = cq.Workplane('XY').circle(stem).extrude(10).faces('>Z').workplane().circle(cap / 2).extrude(2)\n")
    r = verify(src, t)
    assert r.valid and not r.print_pass


def test_overstiff_snap_fails_strength(instances):
    t = _task(instances, "cable_clip")
    d = dict(t.inst.design, wall=4.0, aperture=t.inst.mating["cable_diameter"] * 0.5)
    r = verify(FAMILIES["cable_clip"].program(d), t)
    assert r.valid and not r.strength_pass and r.strength_util > 1.0


def test_magic_numbers_score_low():
    named = FAMILIES["spacer"].program({"outer_diameter": 12.0, "bore_diameter": 5.3, "height": 8.0, "hex": 0.0,
                                        "chamfer": 0.0})
    magic = "import cadquery as cq\nresult = cq.Workplane('XY').circle(6.13).circle(2.67).extrude(8.31)\n"
    assert static_editability(named).score > static_editability(magic).score + 0.3


def test_heldout_agrees_on_good_and_bad(instances):
    t = _task(instances, "spacer")
    good = FAMILIES["spacer"].program(t.inst.design)
    bad = FAMILIES["spacer"].program(dict(t.inst.design, bore_diameter=t.inst.mating["bolt_diameter"] - 0.1))
    hg = verify(good, t, suite="heldout", slicer=False, fea=False, functional_edit=False)
    hb = verify(bad, t, suite="heldout", slicer=False, fea=False, functional_edit=False)
    assert hg.fit_pass and not hb.fit_pass


@pytest.mark.slicer
def test_slicer_runs(instances):
    from spare.verifiers.printability.printability import slicer_available

    if not slicer_available():
        pytest.skip("no slicer")
    t = _task(instances, "spacer")
    r = verify(FAMILIES["spacer"].program(t.inst.design), t, suite="heldout", fea=False, functional_edit=False)
    assert r.print_detail["backend"] == "prusaslicer" and r.print_detail.get("minutes", 0) > 0


_ = np
