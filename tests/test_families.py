import pytest

from spare.common.geom import check_validity
from spare.common.sandbox import run_program_local
from spare.synth.registry import FAMILIES
from spare.verifiers.fit.fit import check_fit
from spare.verifiers.geometry.probes import OCCTProber, run_probes


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_family_design_is_valid_and_fits(name, instances):
    f = FAMILIES[name]
    inst = instances[name]
    for style in range(f.n_styles):
        r = run_program_local(f.program(inst.design, style), trusted=True)
        assert r.ok, r.error
        shape = r.shape()
        assert check_validity(shape).ok
        m = run_program_local(f.mating_program(inst.mating, inst.design), trusted=True)
        assert m.ok, m.error
        fit = check_fit(shape, f.fit_specs(inst.mating, inst.design), m.shape())
        assert fit.passed, [(s.name, s.verdict, s.median_printed_gap) for s in fit.specs]
        pr = run_probes(OCCTProber(shape), f.probes(inst.design))
        assert all(p.ok for p in pr), [(p.name, p.measured, p.target) for p in pr if not p.ok]


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_family_original_and_broken_build(name, instances):
    import numpy as np

    f = FAMILIES[name]
    inst = instances[name]
    assert run_program_local(f.program(inst.original), trusted=True).ok
    src, meta = f.break_program(inst.original, np.random.default_rng(0))
    r = run_program_local(src, trusted=True)
    assert r.ok, r.error
    assert "type" in meta


def test_feature_graph_marks_critical(instances):
    for name, f in FAMILIES.items():
        g = f.feature_graph(instances[name].mating, instances[name].design)
        txt = str(g)
        assert "critical" in txt and "mates_with" in txt, name


def test_splits_are_by_family():
    splits = {f.split for f in FAMILIES.values()}
    assert splits == {"train", "val", "test"}
    assert {f.name for f in FAMILIES.values() if f.split == "test"} == {"hinge_leaf", "rack_clip", "end_cap_rect"}
