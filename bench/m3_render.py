"""M3 on rendered photos: (a) feature-type detection, (b) mating-dimension error from photos of the
counterpart, (c) completion of broken round parts (rotational symmetry) vs intact originals."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time

import numpy as np

from spare.common.provenance import bootstrap_ci, read_jsonl, stamp, write_result

ROUND_BREAK = {"spacer", "knob_dshaft", "cap_round", "wheel_grooved"}
# mating items that stand on the card as vertical cylinders: (mating param, expected detected kind)
MATING_ROUND = {"spacer": "bolt_diameter", "knob_dshaft": "shaft_diameter", "wheel_grooved": "axle_diameter",
                "cap_round": "tube_outer_diameter"}
TYPE_MAP = {"through_hole": "through_hole", "blind_hole": "blind_hole", "blind_d_bore": "d_bore",
            "boss": "boss"}


def gt_types(rec) -> set[str]:
    from spare.synth.registry import get

    g = get(rec["family"]).feature_graph(rec["mating"], rec["original"], rec["original"])
    out = set()
    for f in g["features"]:
        t = TYPE_MAP.get(f["type"])
        if t and f.get("axis", "z") == "z":
            out.add(t)
    return out


def _capture(src, seed, n_views=8):
    from spare.common.geom import shape_to_mesh
    from spare.common.sandbox import run_program_local
    from spare.metrology.calib import calibrate
    from spare.synth.render import render_part_views

    r = run_program_local(src, trusted=True)
    if not r.ok:
        return None
    mesh = shape_to_mesh(r.shape(), tol=0.02)
    cap = render_part_views(mesh, np.random.default_rng(seed), n_views=n_views)
    return cap, calibrate(cap["images"])


def job(args) -> dict:
    try:
        return _job(args)
    except Exception as e:  # noqa: BLE001
        import traceback

        return {"kind": args[0], "uid": args[1]["uid"], "family": args[1]["family"],
                "error": f"{type(e).__name__}: {e}", "where": traceback.format_exc().splitlines()[-3][:200]}


def _job(args) -> dict:
    kind, rec, seed = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from spare.features.completion import complete_round
    from spare.features.infer import snap_standard, standard_kind
    from spare.metrology import reconstruct as R
    from spare.synth.registry import get

    fam = get(rec["family"])
    out = {"kind": kind, "uid": rec["uid"], "family": rec["family"]}
    if kind == "types":
        c = _capture(fam.program(rec["original"], rec["style"]), seed)
        if c is None:
            return {**out, "error": "exec"}
        m, _, _ = R.measure_part(c[0]["images"], c[1], n_boot=0)
        det = set()
        bosses = [ci for ci in m.circles if ci.kind == "boss"]
        body = max(bosses, key=lambda ci: ci.d) if bosses else None  # outer body outline is not a feature
        for ci in m.circles:
            if ci is body:
                continue
            if ci.kind == "hole":
                det.add("d_bore" if ci.flat_offset is not None else "through_hole")
            elif ci.kind == "blind_hole":
                det.add("d_bore" if ci.flat_offset is not None else "blind_hole")
            elif ci.kind == "boss":
                det.add("boss")
        return {**out, "gt": sorted(gt_types(rec)), "det": sorted(det)}
    if kind == "mating":
        q = MATING_ROUND[rec["family"]]
        c = _capture(fam.mating_program(rec["mating"], rec["design"]), seed)
        if c is None:
            return {**out, "error": "exec"}
        m, _, _ = R.measure_part(c[0]["images"], c[1], n_boot=0)
        bosses = [ci for ci in m.circles if ci.kind == "boss"]
        if not bosses:
            return {**out, "param": q, "true": rec["mating"][q], "meas": None}
        b = max(bosses, key=lambda ci: ci.d)
        snapped, ok = snap_standard(b.d, max(b.sigma if np.isfinite(b.sigma) else 0.3, 0.3), standard_kind(q))
        return {**out, "param": q, "true": rec["mating"][q], "meas": b.d, "meas_standard": snapped, "snapped": ok,
                "flat_detected": b.flat_offset is not None}
    if kind == "completion":
        src, info = fam.break_program(rec["original"], np.random.default_rng(seed))
        c = _capture(src, seed)
        if c is None:
            return {**out, "error": "exec"}
        cap, cal = c
        masks = [R.segment(im, v) if v is not None else np.zeros(im.shape[:2], np.uint8)
                 for im, v in zip(cap["images"], cal.views)]
        h0 = R.visual_hull(masks, cal.views, 0.3)
        g = R._grad_images(cap["images"], cal.views)
        z, _ = R.top_height_sweep(cap["images"], cal.views, h0, g)
        h = R.clip_hull(h0, z)
        P2, N2 = R.outline_points(h)
        _, offs = R.refine_outline(P2, N2, z, cal.views, g)
        off = float(np.clip(np.median(offs), -1.5, 0.3)) if len(offs) > 30 else 0.0
        m, _, _ = R.measure_part(cap["images"], cal, n_boot=0, masks=masks)
        true_d = {"spacer": "outer_diameter", "knob_dshaft": "knob_diameter", "cap_round": None,
                  "wheel_grooved": "wheel_diameter"}[rec["family"]]
        td = rec["original"][true_d] if true_d else rec["original"]["inner_diameter"] + 2 * rec["original"]["wall"]
        if rec["family"] == "spacer" and rec["original"].get("hex", 0) > 0.5:
            return {**out, "skip": "hex"}
        comp = complete_round(h, off)
        bosses = [ci for ci in m.circles if ci.kind == "boss"]
        naive = max(m.extent_major, m.extent_minor)
        return {**out, "break": info, "true_d": td, "naive_extent_d": naive,
                "circle_detector_d": max(b.d for b in bosses) if bosses else None,
                "completed_d": comp["diameter"] if comp else None,
                "arc_coverage_deg": comp["arc_coverage_deg"] if comp else None}
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-types", type=int, default=60)
    ap.add_argument("--n-mating", type=int, default=40)
    ap.add_argument("--n-completion", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/m3/render.json")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    recs = read_jsonl("data/benchmarks/test_id.jsonl") + read_jsonl("data/benchmarks/val.jsonl") + \
        read_jsonl("data/benchmarks/test.jsonl")
    jobs = []
    for i in rng.choice(len(recs), a.n_types, replace=False):
        jobs.append(("types", recs[int(i)], int(rng.integers(1 << 30))))
    mr = [r for r in recs if r["family"] in MATING_ROUND]
    for i in rng.choice(len(mr), min(a.n_mating, len(mr)), replace=False):
        jobs.append(("mating", mr[int(i)], int(rng.integers(1 << 30))))
    cr = [r for r in recs if r["family"] in ROUND_BREAK and not (r["family"] == "spacer" and r["original"]["hex"] > 0.5)]
    for i in rng.choice(len(cr), min(a.n_completion, len(cr)), replace=False):
        jobs.append(("completion", cr[int(i)], int(rng.integers(1 << 30))))
    t0 = time.time()
    with mp.get_context("spawn").Pool(max(1, (os.cpu_count() or 4) - 2)) as pool:
        res = pool.map(job, jobs, chunksize=1)
    # (a) feature types: per-type precision/recall over parts
    ty = [r for r in res if r["kind"] == "types" and "gt" in r]
    types = sorted({t for r in ty for t in r["gt"]} | {t for r in ty for t in r["det"]})
    ft = {}
    for t in types:
        tp = sum(t in r["gt"] and t in r["det"] for r in ty)
        fp = sum(t not in r["gt"] and t in r["det"] for r in ty)
        fn = sum(t in r["gt"] and t not in r["det"] for r in ty)
        ft[t] = {"tp": tp, "fp": fp, "fn": fn, "precision": tp / (tp + fp) if tp + fp else None,
                 "recall": tp / (tp + fn) if tp + fn else None}
    exact = [set(r["gt"]) == set(r["det"]) for r in ty]
    # (b) mating
    ma = [r for r in res if r["kind"] == "mating" and r.get("meas") is not None]
    nm = sum(1 for r in res if r["kind"] == "mating")
    e_raw = [abs(r["meas"] - r["true"]) for r in ma]
    e_std = [abs(r["meas_standard"] - r["true"]) for r in ma]
    mating = {"n": nm, "detected": len(ma),
              "abs_err_median_raw": bootstrap_ci(e_raw, stat=np.median) if e_raw else None,
              "abs_err_p90_raw": float(np.quantile(e_raw, 0.9)) if e_raw else None,
              "abs_err_median_with_standard_sizes": bootstrap_ci(e_std, stat=np.median) if e_std else None,
              "abs_err_p90_with_standard_sizes": float(np.quantile(e_std, 0.9)) if e_std else None,
              "frac_within_0.1mm_with_standard_sizes": float(np.mean(np.array(e_std) <= 0.1)) if e_std else None,
              "by_param": {q: float(np.median([abs(r["meas_standard"] - r["true"]) for r in ma if r["param"] == q]))
                           for q in sorted({r["param"] for r in ma})}}
    # (c) completion
    co = [r for r in res if r["kind"] == "completion" and "true_d" in r]

    def err(key):
        v = [abs(r[key] - r["true_d"]) for r in co if r.get(key) is not None]
        return {"n": len(v), "of": len(co), "abs_err_median": float(np.median(v)) if v else None,
                "abs_err_p90": float(np.quantile(v, 0.9)) if v else None}

    completion = {"naive_extent": err("naive_extent_d"), "circle_detector": err("circle_detector_d"),
                  "rotational_completion": err("completed_d")}
    write_result(a.out, {"seconds": round(time.time() - t0, 1),
                         "feature_types": {"per_type": ft, "n_parts": len(ty),
                                           "exact_set_match": float(np.mean(exact)) if exact else None},
                         "mating_from_photos": mating, "completion_round": completion, "rows": res},
                 stamp(a.seed, vars(a)))
    print("types", ft, "exact", np.mean(exact) if exact else None)
    print("mating", mating)
    print("completion", completion)


if __name__ == "__main__":
    main()
