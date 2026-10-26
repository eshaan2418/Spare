"""M1: metrology accuracy and uncertainty calibration on synthetic renders with known geometry.

For sampled benchmark parts, render a capture set of the *original* part on the virtual scale card, run
calibration + visual-hull + sub-pixel circle metrology, and compare against ground truth measured on the
true geometry (bbox height/extents; vertical circular features from the family probes).
Outputs absolute-error distributions by feature type and the empirical coverage of the stated ±1σ/±2σ.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import time

import numpy as np

from spare.common.provenance import bootstrap_ci, read_jsonl, stamp, write_result


def _gt_circles(rec: dict) -> list[dict]:
    from spare.synth.registry import get

    fam = get(rec["family"])
    out = []
    for p in fam.probes(rec["original"]):
        if p.kind == "circle_d" and p.args.get("axis") == 2 and "near" in p.args:
            out.append({"name": p.name, "near": list(p.args["near"]), "d": p.target, "z": p.args.get("offset", 0.0),
                        "which": p.args.get("which", "inner"), "critical": p.critical})
    return out


def measure_one(args) -> dict:
    rec, seed, n_views = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from spare.common.geom import shape_to_mesh
    from spare.common.sandbox import run_program_local
    from spare.metrology.calib import calibrate
    from spare.metrology.reconstruct import measure_part
    from spare.synth.registry import get
    from spare.synth.render import render_part_views

    t0 = time.time()
    fam = get(rec["family"])
    r = run_program_local(fam.program(rec["original"], rec["style"]), trusted=True)
    if not r.ok:
        return {"uid": rec["uid"], "error": r.error}
    shape = r.shape()
    mesh = shape_to_mesh(shape, tol=0.02)
    rng = np.random.default_rng(seed)
    cap = render_part_views(mesh, rng, n_views=n_views)
    cal = calibrate(cap["images"])
    m, hull, _ = measure_part(cap["images"], cal)
    v0 = np.asarray(mesh.vertices, float)
    c, s = math.cos(cap["yaw"]), math.sin(cap["yaw"])
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    shift = cap["verts"][0] - Rz @ v0[0]
    lo, hi = v0.min(0), v0.max(0)
    # GT extents in the measured principal frame: rotate true placed verts into it
    ang = math.radians(m.axis_angle_deg)
    ax = np.array([[math.cos(ang), math.sin(ang)], [-math.sin(ang), math.cos(ang)]])
    pv = cap["verts"][:, :2] @ ax.T
    ext = pv.max(0) - pv.min(0)
    rows = [{"feature": "height", "true": float(hi[2] - lo[2]), "meas": m.height, "sigma": m.sigma.get("height")},
            {"feature": "extent", "true": float(ext[0]), "meas": m.extent_major, "sigma": m.sigma.get("extent_major")},
            {"feature": "extent", "true": float(ext[1]), "meas": m.extent_minor, "sigma": m.sigma.get("extent_minor")}]
    gts = _gt_circles(rec)
    used = set()
    for g in gts:
        cw = Rz @ np.r_[g["near"], 0.0] + shift
        best, bd = None, 1e9
        want = ("hole", "blind_hole") if g["which"] == "inner" else ("boss",)
        for k, ci in enumerate(m.circles):
            if ci.kind not in want:
                continue
            dd = math.hypot(ci.center[0] - cw[0], ci.center[1] - cw[1])
            if dd < bd and k not in used:
                best, bd = k, dd
        if best is not None and bd < max(2.0, 0.5 * g["d"]):
            used.add(best)
            ci = m.circles[best]
            rows.append({"feature": "hole_d" if g["which"] == "inner" else "boss_d", "true": g["d"], "meas": ci.d,
                         "sigma": ci.sigma if math.isfinite(ci.sigma) else None, "name": g["name"],
                         "critical": g["critical"]})
        else:
            rows.append({"feature": "hole_d" if g["which"] == "inner" else "boss_d", "true": g["d"], "meas": None,
                         "sigma": None, "name": g["name"], "missed": True})
    return {"uid": rec["uid"], "family": rec["family"], "rows": rows, "n_views_used": m.n_views_used,
            "calib_rms_px": cal.rms_px, "n_circles": len(m.circles), "seconds": time.time() - t0}


def summarise(results: list[dict]) -> dict:
    out = {}
    feats = sorted({r["feature"] for res in results for r in res.get("rows", [])})
    for f in feats:
        per_part_err, errs, z1, z2, missed, n = [], [], [], [], 0, 0
        for res in results:
            pe = []
            for r in res.get("rows", []):
                if r["feature"] != f:
                    continue
                n += 1
                if r["meas"] is None:
                    missed += 1
                    continue
                e = abs(r["meas"] - r["true"])
                errs.append(e)
                pe.append(e)
                if r["sigma"]:
                    z1.append(e <= r["sigma"])
                    z2.append(e <= 2 * r["sigma"])
            if pe:
                per_part_err.append(float(np.median(pe)))
        med = bootstrap_ci(errs, stat=np.median) if errs else (None, None, None)
        out[f] = {"n": n, "detected": n - missed, "detection_rate": (n - missed) / n if n else None,
                  "abs_err_median": med[0], "abs_err_median_ci95": [med[1], med[2]],
                  "abs_err_p90": float(np.quantile(errs, 0.9)) if errs else None,
                  "coverage_1sigma": float(np.mean(z1)) if z1 else None,
                  "coverage_2sigma": float(np.mean(z2)) if z2 else None, "n_with_sigma": len(z1)}
    return out


def conformal_scales(results: list[dict], q: float = 0.6827) -> dict:
    """Per-feature multiplicative sigma scale so that |err| <= s*sigma at the nominal 1-sigma rate."""
    out = {}
    for f in sorted({r["feature"] for res in results for r in res.get("rows", [])}):
        z = [abs(r["meas"] - r["true"]) / r["sigma"] for res in results for r in res.get("rows", [])
             if r["feature"] == f and r["meas"] is not None and r["sigma"]]
        if len(z) >= 5:
            out[f] = float(np.quantile(z, q))
    return out


def apply_scales(results: list[dict], scales: dict) -> list[dict]:
    out = []
    for res in results:
        rr = dict(res)
        rr["rows"] = [dict(r, sigma=(r["sigma"] * scales.get(r["feature"], 1.0)) if r.get("sigma") else r.get("sigma"))
                      for r in res.get("rows", [])]
        out.append(rr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=90)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--views", type=int, default=8)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default="results/m1/metrology_synth.json")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    calib = read_jsonl("data/benchmarks/test_id.jsonl")  # train families: sigma calibration only
    evals = read_jsonl("data/benchmarks/val.jsonl") + read_jsonl("data/benchmarks/test.jsonl")
    n_c = a.n // 3
    pc = rng.choice(len(calib), size=min(n_c, len(calib)), replace=False)
    pe = rng.choice(len(evals), size=min(a.n - n_c, len(evals)), replace=False)
    picked = [("calib", calib[int(i)]) for i in pc] + [("eval", evals[int(i)]) for i in pe]
    jobs = [(r, a.seed * 100003 + k, a.views) for k, (_, r) in enumerate(picked)]
    t0 = time.time()
    w = a.workers or max(1, (os.cpu_count() or 4) - 2)
    with mp.get_context("spawn").Pool(w) as pool:
        results = pool.map(measure_one, jobs, chunksize=1)
    for (role, _), res in zip(picked, results):
        res["role"] = role
    cal_res = [r for r in results if r["role"] == "calib"]
    ev_res = [r for r in results if r["role"] == "eval"]
    scales = conformal_scales(cal_res)
    payload = {"n_parts": len(results), "n_calib": len(cal_res), "n_eval": len(ev_res),
               "seconds": round(time.time() - t0, 1),
               "summary_all_raw": summarise(results),
               "summary_eval_raw": summarise(ev_res),
               "sigma_scales_from_calib": scales,
               "summary_eval_calibrated": summarise(apply_scales(ev_res, scales)),
               "by_family_height_abs_err_median": {
                   f: float(np.median([abs(r["meas"] - r["true"]) for res in results if res.get("family") == f
                                       for r in res["rows"] if r["feature"] == "height" and r["meas"] is not None]))
                   for f in sorted({res.get("family") for res in results if res.get("family")})},
               "parts": results}
    write_result(a.out, payload, stamp(a.seed, vars(a)))
    from spare.common.provenance import ROOT
    (ROOT / "spare" / "metrology" / "sigma_scale.json").write_text(json.dumps(scales, indent=1))
    for key in ("summary_eval_raw", "summary_eval_calibrated"):
        print(key)
        for f, s in payload[key].items():
            print(" ", f, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items() if k != "abs_err_median_ci95"})


if __name__ == "__main__":
    main()
