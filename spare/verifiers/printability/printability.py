"""Printability verifiers.

Primary (reward): geometric analysis of the tessellated part in its canonical
print orientation (Z up): build volume, bed contact, unsupported overhang area,
thin walls, and rough time/material estimates.
Held-out: PrusaSlicer CLI slices with supports off and on (support filament =
difference), plus slicer errors.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

BUILD_VOLUME = (220.0, 220.0, 250.0)
OVERHANG_DEG = 45.0
BRIDGE_DROP = 8.0  # mm  # from vertical; steeper down-facing surfaces need support
MIN_WALL = 0.75      # < ~2 extrusion widths for a 0.4 nozzle
SUPPORT_FRAC_MAX = 0.15  # held-out: >15% of filament as auto-support (45 deg) = "needs significant support"
REPO = Path(__file__).resolve().parents[3]
PRUSA = os.environ.get("PRUSASLICER", "/Applications/PrusaSlicer.app/Contents/MacOS/PrusaSlicer")


@dataclass
class PrintResult:
    passed: bool
    score: float
    metrics: dict = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    backend: str = "geometric"


def analyse_geometric(mesh, n_wall_samples: int = 300, seed: int = 0, shape=None) -> PrintResult:
    reasons = []
    lo, hi = mesh.bounds
    size = hi - lo
    fits = bool(np.all(size <= np.array(BUILD_VOLUME)))
    if not fits:
        reasons.append(f"exceeds build volume {size.round(1).tolist()}")
    n = mesh.face_normals
    a = mesh.area_faces
    c = mesh.triangles_center
    zmin = lo[2]
    on_bed = (np.abs(c[:, 2] - zmin) < 0.05) & (n[:, 2] < -0.99)
    bed_area = float(a[on_bed].sum())
    footprint = float(size[0] * size[1])
    if bed_area < max(8.0, 0.03 * footprint):
        reasons.append(f"small bed contact {bed_area:.1f} mm2")
    down = n[:, 2] < -np.cos(np.radians(OVERHANG_DEG))
    overhang = down & ~on_bed
    # short bridges (horizontal down faces over small spans) print fine: tolerate small total area
    oh_area = float(a[overhang].sum())
    # overhangs that arch over the part's own material within BRIDGE_DROP (tops of small horizontal holes,
    # short bridges) print unsupported; only count facets with nothing below within that distance
    if oh_area > 0 and shape is not None:
        from OCP.gp import gp_Dir, gp_Lin, gp_Pnt

        from spare.common.geom import OCCTQuery

        q = OCCTQuery(shape)
        idx = np.where(overhang)[0]
        rng0 = np.random.default_rng(seed + 1)
        pick = rng0.choice(idx, size=min(200, len(idx)), replace=False, p=a[idx] / a[idx].sum())
        unsupported = 0
        for k in pick:
            o = c[k] - np.array([0, 0, 0.05])
            q._isect.Perform(gp_Lin(gp_Pnt(*map(float, o)), gp_Dir(0, 0, -1)), 0.0, BRIDGE_DROP)
            if not q._isect.IsDone() or q._isect.NbPnt() == 0:
                unsupported += 1
        oh_area = oh_area * unsupported / len(pick)
    total = float(a.sum())
    oh_allow = 15.0 + 0.02 * total
    if oh_area > oh_allow:
        reasons.append(f"overhang needs support ({oh_area:.0f} mm2)")
    # thin walls: inward rays from random surface samples
    rng = np.random.default_rng(seed)
    try:
        pts, fi = mesh.sample(n_wall_samples, return_index=True, seed=int(rng.integers(1 << 30)))
    except TypeError:
        pts, fi = mesh.sample(n_wall_samples, return_index=True)
    dirs = -n[fi]
    th = np.full(len(pts), np.inf)
    if shape is not None:
        from OCP.gp import gp_Dir, gp_Lin, gp_Pnt

        from spare.common.geom import OCCTQuery

        q = OCCTQuery(shape)
        isect = q._isect
        for i, (p, d) in enumerate(zip(pts, dirs)):
            o = p - d * 0.1  # start just outside: tessellation chords can sit inside or outside the B-rep
            isect.Perform(gp_Lin(gp_Pnt(*map(float, o)), gp_Dir(*map(float, d))), 0.0, 10.0)
            if not isect.IsDone() or isect.NbPnt() < 2:
                continue
            hs = sorted(isect.WParameter(k) for k in range(1, isect.NbPnt() + 1))
            th[i] = hs[1] - hs[0]
    else:
        origins = pts + dirs * 1e-4
        try:
            locs, idx, _ = mesh.ray.intersects_location(origins, dirs, multiple_hits=False)
            th[idx] = np.linalg.norm(locs - origins[idx], axis=1)
        except Exception:  # noqa: BLE001
            pass
    thin_frac = float(np.mean(th < MIN_WALL))
    if thin_frac > 0.03:
        reasons.append(f"thin walls ({thin_frac:.0%} of surface < {MIN_WALL} mm)")
    vol = float(mesh.volume)
    if vol < 5.0:
        reasons.append("part too small to print reliably")
    grams = vol / 1000.0 * 1.24 * 0.55  # ~55% effective fill (perimeters + 25% infill), PLA
    minutes = 2.0 + vol / 1000.0 * 6.0 + size[2] / 0.2 * 0.05
    passed = not reasons
    s_oh = float(np.clip(1 - max(oh_area - oh_allow, 0) / max(total * 0.2, 1e-6), 0, 1))
    s_bed = float(np.clip(bed_area / max(8.0, 0.03 * footprint), 0, 1))
    s_thin = float(np.clip(1 - max(thin_frac - 0.03, 0) * 10, 0, 1))
    score = float((s_oh + s_bed + s_thin) / 3.0) * (1.0 if fits and vol >= 5.0 else 0.0)
    return PrintResult(passed, score, {
        "size_mm": size.round(2).tolist(), "bed_area_mm2": round(bed_area, 1), "overhang_area_mm2": round(oh_area, 1),
        "thin_wall_frac": round(thin_frac, 4), "volume_mm3": round(vol, 1), "est_grams": round(grams, 2),
        "est_minutes": round(float(minutes), 1)}, reasons, "geometric")


def _parse_gcode(path: Path) -> dict:
    txt = path.read_text(errors="ignore")[-20000:]
    m = re.search(r"estimated printing time \(normal mode\) = (.+)", txt)
    mm = re.search(r"filament used \[mm\] = ([\d.]+)", txt)
    secs = 0.0
    if m:
        for val, unit in re.findall(r"(\d+)([dhms])", m.group(1)):
            secs += int(val) * {"d": 86400, "h": 3600, "m": 60, "s": 1}[unit]
    return {"seconds": secs, "filament_mm": float(mm.group(1)) if mm else 0.0}


def slicer_available() -> bool:
    return Path(PRUSA).exists() or shutil.which("prusa-slicer") is not None


def analyse_slicer(mesh, profile: str = "generic_pla_0.4", timeout: int = 120) -> PrintResult:
    if not slicer_available():
        return PrintResult(False, 0.0, {}, ["slicer unavailable"], "prusaslicer")
    ini = REPO / "configs" / "slicer" / f"{profile}.ini"
    with tempfile.TemporaryDirectory() as td_:
        td = Path(td_)
        stl = td / "part.stl"
        mesh.export(stl)
        res: dict = {}
        for sup in (0, 1):
            out = td / f"out{sup}.gcode"
            cmd = [PRUSA, "--datadir", str(td / "dd"), "--load", str(ini), "--support-material" if sup else "--no-support-material",
                   "--export-gcode", "-o", str(out), str(stl)]
            if sup:
                cmd[6:6] = ["--support-material-auto"]
            try:
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            except subprocess.TimeoutExpired:
                return PrintResult(False, 0.0, {}, ["slicer timeout"], "prusaslicer")
            if p.returncode != 0 or not out.exists():
                err = (p.stderr or p.stdout).strip().splitlines()[-1:] or ["unknown"]
                return PrintResult(False, 0.0, {"stderr": err}, [f"slice failed: {err[0][:120]}"], "prusaslicer")
            res[sup] = _parse_gcode(out)
            res[f"log{sup}"] = (p.stdout + p.stderr)[-2000:]
    fil0, fil1 = res[0]["filament_mm"], res[1]["filament_mm"]
    support_frac = max(fil1 - fil0, 0.0) / max(fil1, 1e-6)
    grams = fil0 * np.pi * (1.75 / 2) ** 2 / 1000.0 * 1.24
    reasons = []
    if support_frac > SUPPORT_FRAC_MAX:
        reasons.append(f"needs support ({support_frac:.0%} of filament)")
    for k in ("log0",):
        if re.search(r"(empty layer|Detected print stability issues|failed)", res[k], re.I):
            m_ = re.search(r"(empty layer[^\n]*|Detected print stability issues:?\s*[^\n]*\n?[^\n]*|failed[^\n]*)", res[k], re.I)
            reasons.append("slicer warning: " + (m_.group(0).replace("\n", " ").strip()[:160] if m_ else "?"))
    warnings = [r for r in reasons if r.startswith("slicer warning")]
    reasons = [r for r in reasons if not r.startswith("slicer warning")]
    passed = not reasons  # stability warnings (e.g. "low bed adhesion" -> add brim) are a soft penalty
    score = float(np.clip(1 - max(support_frac - 0.05, 0) * 4, 0, 1)) * (1.0 if passed else 0.7) - 0.1 * bool(warnings)
    score = max(score, 0.0)
    return PrintResult(passed, score, {"minutes": round(res[0]["seconds"] / 60, 1), "filament_mm": fil0,
                                       "grams": round(float(grams), 2), "support_frac": round(support_frac, 3),
                                       "warnings": warnings},
                       reasons, "prusaslicer")
