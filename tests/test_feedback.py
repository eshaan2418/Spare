import numpy as np

from spare.feedback.compensation import CompensationModel, Observation
from spare.feedback.fit_form import load, save
from spare.feedback.policy import correct
from spare.feedback.report import FitReport, parse_text
from spare.features.tolerance import FIT_BANDS
from spare.synth.registry import get


def test_parse_text():
    r = parse_text("too tight on the shaft", "p1", 1)
    assert (r.outcome, r.location) == ("too_tight", "shaft")
    assert parse_text("snap broke at the root").outcome == "broke"
    assert parse_text("wobbles on the axle").outcome == "too_loose"
    assert parse_text("clicks in, perfect").outcome == "fits"


def test_fit_form_roundtrip(tmp_path):
    save(FitReport("p9", 1, "too_loose", "bolt"), tmp_path)
    save(FitReport("p9", 2, "fits"), tmp_path)
    rs = load("p9", tmp_path)
    assert [r.outcome for r in rs] == ["too_loose", "fits"] and rs[0].evidence == "physical"


def test_policy_opens_tight_bore():
    fam = get("spacer")
    inst = fam.instance(np.random.default_rng(0), "t")
    src = fam.program(inst.design)
    specs = fam.fit_specs(inst.mating, inst.design)
    c = correct(src, FitReport("x", 1, "too_tight", specs[0].location), specs)
    assert c.applied and c.edits[0].param == specs[0].param
    e = c.edits[0]
    assert (e.new - e.old) * specs[0].gap_sign > 0  # gap opens
    c2 = correct(src, FitReport("x", 1, "too_loose", specs[0].location), specs)
    assert (c2.edits[0].new - c2.edits[0].old) * specs[0].gap_sign < 0


def test_compensation_recovers_bias():
    rng = np.random.default_rng(0)
    m = CompensationModel(noise=0.03)
    true_b = 0.12  # printer grows holes 0.12 mm/side more than the default model
    from spare.features.tolerance import printed_deviation
    for _ in range(25):
        g = rng.uniform(0.0, 0.6)
        printed = g - printed_deviation("hole", 5.0) - true_b + rng.normal(0, 0.03)
        lo, hi = FIT_BANDS["clearance"]
        out = "fits" if lo <= printed <= hi else "too_tight" if printed < lo else "too_loose"
        m.add(Observation("hole", 5.0, "clearance", g, out))
    b = m.fit()["hole"]
    assert abs(b - true_b) < 0.05
