"""Multi-view metrology on the scale card.

1. segment(): part mask by known-background differencing (the card's appearance in
   each view is predicted from the calibrated pose), with illumination
   normalisation and a shadow test (darker but achromatic -> background).
2. visual_hull(): voxel carving in the metric card frame (k-of-n consistency).
3. circles_from_hull() + refine_circle(): candidate round features from the hull,
   refined with sub-pixel image edges back-projected onto the feature's plane.
4. bootstrap over views for 1-sigma uncertainty of every measurement.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from spare.capture.card import CARD
from spare.metrology.calib import CalibratedView, Calibration
from spare.synth.render import PAGE_PPM, page_texture
from spare.verifiers.geometry.probes import _robust_circle


# ----------------------------------------------------------------------------- segmentation
def expected_card(view: CalibratedView) -> tuple[np.ndarray, np.ndarray]:
    ox, oy = CARD.origin_mm
    A = np.array([[1.0, 0, -ox], [0, -1.0, oy], [0, 0, 1.0]])
    S = np.diag([1 / PAGE_PPM, 1 / PAGE_PPM, 1.0])
    P = view.K @ np.c_[view.R[:, 0], view.R[:, 1], view.t]
    H = P @ A @ S
    w, h = view.size
    tex = page_texture().astype(np.float32)
    E = cv2.warpPerspective(tex, H, (w, h), flags=cv2.INTER_LINEAR, borderValue=-1)
    return E, E >= 0


def stage_polygon_px(view: CalibratedView, margin: float = 12.0) -> np.ndarray:
    x0, y0, x1, y1 = CARD.stage_mm
    pts = np.array([[x0 - margin, -(y0 - margin), 0], [x1 + margin, -(y0 - margin), 0],
                    [x1 + margin, -(y1 + margin), 0], [x0 - margin, -(y1 + margin), 0]])
    return view.project(pts).astype(np.int32)


def segment(img: np.ndarray, view: CalibratedView, diff_thr: float = 0.28, chroma_thr: float = 22.0,
            shadow_lo: float = 0.40) -> np.ndarray:
    E, card = expected_card(view)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # match blur of the real optics a little
    E = cv2.GaussianBlur(np.where(card, E, 0), (0, 0), 0.8)
    roi = np.zeros(gray.shape, np.uint8)
    cv2.fillConvexPoly(roi, stage_polygon_px(view), 1)
    valid = card & (roi > 0)
    ratio = (gray + 4) / (E + 4)
    # illumination gain field: robust large-scale median of the ratio over bright card pixels
    g = np.where(valid & (E > 100), ratio, np.nan).astype(np.float32)
    small = cv2.resize(np.nan_to_num(g, nan=1.0), (64, 48), interpolation=cv2.INTER_AREA)
    gain = cv2.resize(cv2.medianBlur(small, 5), (gray.shape[1], gray.shape[0]))
    pred = gain * E
    d = np.abs(gray - pred) / (pred + 25.0)
    b, gch, r = [img[..., i].astype(np.float32) for i in range(3)]
    chroma = np.max(np.stack([np.abs(b - gch), np.abs(gch - r), np.abs(b - r)]), 0)
    rel = gray / (pred + 1e-3)
    shadow_like = (rel > shadow_lo) & (rel < 0.95) & (chroma < chroma_thr)
    m = valid & (((d > diff_thr) & ~shadow_like) | (chroma > chroma_thr))
    m = m.astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    # keep components that touch the central stage region
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    keep = np.zeros_like(m)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= 80:
            keep[lab == i] = 255
    return keep


# ----------------------------------------------------------------------------- visual hull
@dataclass
class Hull:
    occ: np.ndarray  # (nx, ny, nz) bool
    origin: np.ndarray  # world xyz of voxel (0,0,0) centre
    voxel: float

    def points(self) -> np.ndarray:
        idx = np.argwhere(self.occ)
        return self.origin + idx * self.voxel

    @property
    def height(self) -> float:
        idx = np.argwhere(self.occ)
        return float((idx[:, 2].max() + 0.5) * self.voxel + self.origin[2] - 0.0) if len(idx) else 0.0


def _footprint_roi(masks, views, z_max=60.0, margin=4.0) -> tuple[np.ndarray, np.ndarray]:
    """XY box from back-projecting mask pixels of top-most views onto z=0 and z=z_max/2 planes."""
    pts = []
    order = sorted([i for i, v in enumerate(views) if v is not None], key=lambda i: views[i].R[2, 2])
    for i in order[:3]:  # most top-down views (camera z axis pointing down = R[2,2] ~ -1)
        v = views[i]
        ys, xs = np.nonzero(masks[i][::4, ::4])
        if len(xs) == 0:
            continue
        uv = np.c_[xs * 4, ys * 4].astype(np.float64)
        for zp in (0.0, 10.0):
            pts.append(backproject_to_plane(uv, v, zp))
    P = np.vstack(pts)
    lo, hi = np.percentile(P[:, :2], 0.5, axis=0) - margin, np.percentile(P[:, :2], 99.5, axis=0) + margin
    return np.r_[lo, 0.0], np.r_[hi, z_max]


def backproject_to_plane(uv: np.ndarray, v: CalibratedView, z: float) -> np.ndarray:
    pts = cv2.undistortPoints(np.asarray(uv, np.float64).reshape(-1, 1, 2), v.K, v.dist).reshape(-1, 2)
    rays_c = np.c_[pts, np.ones(len(pts))]
    rays_w = rays_c @ v.R  # R^T * d for row vectors
    C = v.center
    s = (z - C[2]) / rays_w[:, 2]
    return C[None] + s[:, None] * rays_w


def visual_hull(masks: list[np.ndarray], views: list[CalibratedView | None], voxel: float = 0.3,
                z_max: float = 60.0, allow_miss: int = 1, view_idx: list[int] | None = None) -> Hull:
    idx = [i for i in (view_idx if view_idx is not None else range(len(views))) if views[i] is not None]
    lo, hi = _footprint_roi(masks, views, z_max)
    nx, ny, nz = (np.ceil((hi - lo) / voxel)).astype(int)
    nz = max(nz, 1)
    xs = lo[0] + (np.arange(nx) + 0.5) * voxel
    ys = lo[1] + (np.arange(ny) + 0.5) * voxel
    zs = lo[2] + (np.arange(nz) + 0.5) * voxel
    # coarse-to-fine: carve columns at full XY res, z in slabs
    votes_out = np.zeros((nx, ny, nz), np.int16)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    for k, z in enumerate(zs):
        P = np.c_[X.ravel(), Y.ravel(), np.full(X.size, z)]
        for i in idx:
            v = views[i]
            assert v is not None
            uv = project_fast(v, P)
            h, w = masks[i].shape
            u = np.round(uv[:, 0]).astype(int)
            vv = np.round(uv[:, 1]).astype(int)
            inside = (u >= 0) & (u < w) & (vv >= 0) & (vv < h)
            out = np.ones(len(P), bool)
            out[inside] = masks[i][vv[inside], u[inside]] == 0
            out[~inside] = False  # unseen -> no vote
            votes_out[:, :, k] += out.reshape(nx, ny)
    occ = votes_out <= allow_miss
    # remove voxels that are below z=0 or isolated
    return Hull(occ, np.array([xs[0], ys[0], zs[0]]), voxel)


def project_fast(v: CalibratedView, P: np.ndarray) -> np.ndarray:
    Xc = P @ v.R.T + v.t
    x = Xc[:, 0] / Xc[:, 2]
    y = Xc[:, 1] / Xc[:, 2]
    k1, k2 = v.dist.ravel()[:2] if v.dist.size >= 2 else (0.0, 0.0)
    r2 = x * x + y * y
    f = 1 + k1 * r2 + k2 * r2 * r2
    return np.c_[v.K[0, 0] * x * f + v.K[0, 2], v.K[1, 1] * y * f + v.K[1, 2]]


# ----------------------------------------------------------------------------- measurements
@dataclass
class Circle:
    kind: str  # hole | boss
    center: list[float]
    z: float
    d: float
    sigma: float = float("nan")
    n_pts: int = 0
    n_views: int = 0
    residual: float = float("nan")
    flat_offset: float | None = None  # distance from centre to a straight flat (D-shape), if detected


@dataclass
class PartMeasurement:
    height: float
    extent_major: float
    extent_minor: float
    axis_angle_deg: float
    footprint_area: float
    circles: list[Circle] = field(default_factory=list)
    sigma: dict = field(default_factory=dict)
    n_views_used: int = 0
    calib_rms_px: float = float("nan")
    notes: list[str] = field(default_factory=list)

    def to_dict(self):
        d = asdict(self)
        return d


def footprint_image(h: Hull) -> np.ndarray:
    return (h.occ.any(axis=2)).astype(np.uint8)


def top_height_map(h: Hull) -> np.ndarray:
    idx = np.where(h.occ.any(2), h.occ.shape[2] - 1 - np.argmax(h.occ[:, :, ::-1], axis=2), -1)
    return np.where(idx >= 0, h.origin[2] + (idx + 0.5) * h.voxel, np.nan)


def circles_from_hull(h: Hull, min_d: float = 1.5) -> list[Circle]:
    """Through-holes = inner contours of the footprint; blind bores = round pits in the top height map;
    bosses/outer = outer contour if round."""
    fp = footprint_image(h)
    out: list[Circle] = []
    cnts, hier = cv2.findContours(fp.T.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)  # transpose: rows=y
    if hier is None:
        return out
    hm = top_height_map(h)
    for c, hh in zip(cnts, hier[0]):
        pts = c.reshape(-1, 2).astype(float)  # (x_idx, y_idx) since transposed
        if len(pts) < 12:
            continue
        P = h.origin[:2] + pts * h.voxel
        cx, cy, r = _robust_circle(P)
        if not np.isfinite(r) or 2 * r < min_d:
            continue
        res = float(np.median(np.abs(np.hypot(P[:, 0] - cx, P[:, 1] - cy) - r)))
        is_inner = hh[3] >= 0
        if res > max(0.12 * r, 0.6):
            continue
        # plane for refinement: local top surface height around the contour
        ix = np.clip(pts[:, 0].astype(int), 0, hm.shape[0] - 1)
        iy = np.clip(pts[:, 1].astype(int), 0, hm.shape[1] - 1)
        zt = float(np.nanmedian(hm[ix, iy])) if np.isfinite(hm[ix, iy]).any() else h.height
        out.append(Circle("hole" if is_inner else "boss", [float(cx), float(cy)], zt, 2 * r, n_pts=len(P),
                          residual=res))
    # blind bores: pits in the height map (top surface drops by > 1.5 mm inside a round region)
    Hmax = np.nanmax(hm) if np.isfinite(hm).any() else 0
    pit = (np.nan_to_num(hm, nan=-1) < Hmax - 1.5) & (fp > 0)
    pit = cv2.morphologyEx(pit.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts2, _ = cv2.findContours(pit.T.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in cnts2:
        pts = c.reshape(-1, 2).astype(float)
        if len(pts) < 12:
            continue
        P = h.origin[:2] + pts * h.voxel
        cx, cy, r = _robust_circle(P)
        if np.isfinite(r) and 2 * r >= min_d:
            res = float(np.median(np.abs(np.hypot(P[:, 0] - cx, P[:, 1] - cy) - r)))
            if res <= max(0.15 * r, 0.6) and not any(np.hypot(cx - o.center[0], cy - o.center[1]) < 0.5 * r for o in out):
                out.append(Circle("blind_hole", [float(cx), float(cy)], float(Hmax), 2 * r, n_pts=len(P), residual=res))
    return out


def _edge_along(gray_f: np.ndarray, gx: np.ndarray, gy: np.ndarray, p: np.ndarray, n: np.ndarray, search: float):
    """Sub-pixel position of the strongest gradient along normal n through p (parabolic peak)."""
    ts = np.arange(-search, search + 0.5, 0.5)
    q = p[None] + ts[:, None] * n[None]
    h, w = gx.shape
    if (q[:, 0] < 1).any() or (q[:, 1] < 1).any() or (q[:, 0] >= w - 2).any() or (q[:, 1] >= h - 2).any():
        return None
    mx = cv2.remap(gx, q[:, 0].astype(np.float32)[None], q[:, 1].astype(np.float32)[None], cv2.INTER_LINEAR)[0]
    my = cv2.remap(gy, q[:, 0].astype(np.float32)[None], q[:, 1].astype(np.float32)[None], cv2.INTER_LINEAR)[0]
    g = np.abs(mx * n[0] + my * n[1])
    k = int(np.argmax(g))
    if k == 0 or k == len(g) - 1 or g[k] < 6.0:
        return None
    a, b, c = g[k - 1], g[k], g[k + 1]
    den = a - 2 * b + c
    off = 0.5 * (a - c) / den if abs(den) > 1e-9 else 0.0
    return p + (ts[k] + off * 0.5) * n


def _grad_images(images, views):
    out = {}
    for i, v in enumerate(views):
        if v is None:
            continue
        gray = cv2.GaussianBlur(cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.2)
        out[i] = (cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    return out


def outline_points(h: Hull, step: float = 0.6, fp: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Outer + inner footprint contour points (world XY) with outward 2D normals, resampled every ``step`` mm."""
    fp = footprint_image(h) if fp is None else fp.astype(np.uint8)
    cnts, _ = cv2.findContours(fp.T.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    P, N = [], []
    for c in cnts:
        q = h.origin[:2] + c.reshape(-1, 2).astype(float) * h.voxel
        if len(q) < 8:
            continue
        k = max(1, int(round(step / h.voxel)))
        q = q[::k]
        t = np.roll(q, -1, 0) - np.roll(q, 1, 0)
        n = np.c_[t[:, 1], -t[:, 0]]
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
        # orient outward: a step along +n must leave the footprint
        probe = np.round((q + 1.5 * h.voxel * n - h.origin[:2]) / h.voxel).astype(int)
        ins = np.zeros(len(q), bool)
        okp = (probe[:, 0] >= 0) & (probe[:, 1] >= 0) & (probe[:, 0] < fp.shape[0]) & (probe[:, 1] < fp.shape[1])
        ins[okp] = fp[probe[okp, 0], probe[okp, 1]] > 0
        if ins.mean() > 0.5:
            n = -n
        P.append(q)
        N.append(n)
    if not P:
        return np.zeros((0, 2)), np.zeros((0, 2))
    return np.vstack(P), np.vstack(N)


def edge_score_at_height(P2: np.ndarray, N2: np.ndarray, z: float, views, grads, min_cam_z: float = 0.0) -> float:
    """Mean directional image gradient at the projection of the outline lifted to height z."""
    scores = []
    P3 = np.c_[P2, np.full(len(P2), z)]
    Q3 = np.c_[P2 + 0.5 * N2, np.full(len(P2), z)]
    for i, (gx, gy) in grads.items():
        v = views[i]
        if v.center[2] < z + min_cam_z:
            continue
        a = project_fast(v, P3)
        b = project_fast(v, Q3)
        d = b - a
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        hh, ww = gx.shape
        ok = (a[:, 0] > 1) & (a[:, 1] > 1) & (a[:, 0] < ww - 2) & (a[:, 1] < hh - 2)
        if ok.sum() < 10:
            continue
        mx = cv2.remap(gx, a[ok, 0].astype(np.float32)[None], a[ok, 1].astype(np.float32)[None], cv2.INTER_LINEAR)[0]
        my = cv2.remap(gy, a[ok, 0].astype(np.float32)[None], a[ok, 1].astype(np.float32)[None], cv2.INTER_LINEAR)[0]
        g = np.abs(mx * d[ok, 0] + my * d[ok, 1])
        scores.append(float(np.mean(np.minimum(g, 120.0))))
    return float(np.mean(scores)) if scores else 0.0


def _sweep_outline(P2, N2, views, grads, z_lo, z_hi, step=0.1, rel=0.6):
    zs = np.arange(z_lo, z_hi + 0.5 * step, step)
    if len(zs) < 5 or len(P2) < 10:
        return None, 0.0, np.zeros((0, 2))
    sc = np.array([edge_score_at_height(P2, N2, z, views, grads) for z in zs])
    sc_s = np.convolve(sc, np.ones(3) / 3, mode="same")
    sc_s[0], sc_s[-1] = sc[0], sc[-1]
    thr = sc_s.max() * rel
    k_best = int(np.argmax(sc_s))
    for k in range(len(zs) - 2, 0, -1):
        if sc_s[k] >= thr and sc_s[k] >= sc_s[k - 1] and sc_s[k] >= sc_s[k + 1]:
            k_best = k
            break
    k = int(np.clip(k_best, 1, len(zs) - 2))
    a, b, c = sc_s[k - 1], sc_s[k], sc_s[k + 1]
    den = a - 2 * b + c
    off = 0.5 * (a - c) / den if abs(den) > 1e-9 else 0.0
    contrast = float(sc_s[k] / (np.median(sc_s) + 1e-6))
    return float(zs[k] + np.clip(off, -1, 1) * step), contrast, np.c_[zs, sc_s]


def top_height_sweep(images, views, h: Hull, grads=None, z_min: float = 0.8, step: float = 0.1,
                     rel: float = 0.6, view_idx: list[int] | None = None) -> tuple[float, np.ndarray]:
    """Plane-sweep the footprint outline in height. The visual hull cannot bound flat tops without
    near-horizontal views (it inflates by ~r*tan(elevation)), but the top outline aligns with image edges in
    every view only at its true height. Uses the full-footprint outline: accurate for single-level parts;
    raised features much smaller than the footprint (tabs, knuckles) are under-estimated -- a documented
    limitation (results/m1); per-level sweeps were tried and rejected because hull level-set contours
    produce spurious edge peaks. Returns (height, score curve)."""
    grads = grads if grads is not None else _grad_images(images, views)
    if view_idx is not None:
        grads = {k: g for k, g in grads.items() if k in set(view_idx)}
    P2, N2 = outline_points(h)
    z, _, curve = _sweep_outline(P2, N2, views, grads, z_min, h.height, step, rel)
    if z is None:
        return h.height, np.zeros(0)
    return z, curve


def refine_outline(P2: np.ndarray, N2: np.ndarray, z: float, views, grads, search_px: float = 5.0
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Sub-pixel outline points on the plane z, pooled over views (world XY), and each point's signed
    offset (mm) from the hull outline along the outward normal (negative = hull was inflated)."""
    out, offs = [], []
    P3 = np.c_[P2, np.full(len(P2), z)]
    Q3 = np.c_[P2 + 0.5 * N2, np.full(len(P2), z)]
    for i, (gx, gy) in grads.items():
        v = views[i]
        if v.center[2] < z + 20:
            continue
        a = project_fast(v, P3)
        b = project_fast(v, Q3)
        pts, src = [], []
        for k, (p, q) in enumerate(zip(a, b)):
            n = q - p
            nn = np.linalg.norm(n)
            if nn < 1e-6:
                continue
            e = _edge_along(None, gx, gy, p, n / nn, search_px)
            if e is not None:
                pts.append(e)
                src.append(k)
        if len(pts) >= 10:
            W = backproject_to_plane(np.array(pts), v, z)[:, :2]
            out.append(W)
            offs.append(((W - P2[src]) * N2[src]).sum(1))
    if not out:
        return np.zeros((0, 2)), np.zeros(0)
    return np.vstack(out), np.concatenate(offs)

def refine_circle(c: Circle, images, views, view_idx=None, n_samples: int = 72, search_px: float = 5.0) -> Circle:
    """Refine centre/diameter using sub-pixel edges on the circle's plane z=c.z in every view that
    sees it; returns edge points pooled across views (stored for bootstrap)."""
    idx = [i for i in (view_idx if view_idx is not None else range(len(views))) if views[i] is not None]
    th = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    r = c.d / 2
    P3 = np.c_[c.center[0] + r * np.cos(th), c.center[1] + r * np.sin(th), np.full(n_samples, c.z)]
    pooled, per_view = [], []
    for i in idx:
        v = views[i]
        # only views from above the plane, roughly facing it
        if v.center[2] < c.z + 20:
            continue
        gray = cv2.GaussianBlur(cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.0)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        uv = v.project(P3)
        ctr = v.project(np.array([[c.center[0], c.center[1], c.z]]))[0]
        pts = []
        for p in uv:
            n = p - ctr
            nn = np.linalg.norm(n)
            if nn < 3:
                continue
            e = _edge_along(gray, gx, gy, p, n / nn, search_px)
            if e is not None:
                pts.append(e)
        if len(pts) >= n_samples // 3:
            W = backproject_to_plane(np.array(pts), v, c.z)[:, :2]
            per_view.append(W)
            pooled.append(W)
    if not pooled:
        return c
    Pall = np.vstack(pooled)
    cx, cy, rr = _robust_circle(Pall)
    if not np.isfinite(rr) or abs(2 * rr - c.d) > max(1.0, 0.3 * c.d):
        return c
    res = np.hypot(Pall[:, 0] - cx, Pall[:, 1] - cy) - rr
    out = Circle(c.kind, [float(cx), float(cy)], c.z, float(2 * rr), n_pts=len(Pall), n_views=len(per_view),
                 residual=float(np.median(np.abs(res))))
    # bootstrap over views (+ points) for sigma
    rng = np.random.default_rng(0)
    ds = []
    for _ in range(60):
        sel = rng.integers(0, len(per_view), len(per_view))
        Q = np.vstack([per_view[j][rng.integers(0, len(per_view[j]), len(per_view[j]))] for j in sel])
        ds.append(2 * _robust_circle(Q)[2])
    out.sigma = float(np.nanstd(ds)) if len(per_view) > 1 else float(np.nanstd(ds) + 0.05)
    out.flat_offset = detect_flat(Pall, cx, cy, rr)
    return out


def detect_flat(P: np.ndarray, cx: float, cy: float, r: float, min_span_deg: float = 50.0) -> float | None:
    """D-flat: a contiguous angular span where edge points sit well inside the circle and lie on a line."""
    ang = np.degrees(np.arctan2(P[:, 1] - cy, P[:, 0] - cx)) % 360
    rad = np.hypot(P[:, 0] - cx, P[:, 1] - cy)
    bins = np.arange(0, 361, 10)
    inside = np.zeros(len(bins) - 1, bool)
    for k in range(len(bins) - 1):
        m = (ang >= bins[k]) & (ang < bins[k + 1])
        if m.sum() >= 2:
            inside[k] = np.median(rad[m]) < r - max(0.08 * r, 0.15)
    if not inside.any():
        return None
    # longest circular run
    run, best, start, bstart = 0, 0, 0, 0
    arr = np.r_[inside, inside]
    for k, f in enumerate(arr):
        if f:
            if run == 0:
                start = k
            run += 1
            if run > best:
                best, bstart = run, start
        else:
            run = 0
    best = min(best, len(inside))
    if best * 10 < min_span_deg:
        return None
    mid = np.radians((bstart + best / 2) * 10)
    m = np.zeros(len(P), bool)
    for k in range(bstart, bstart + best):
        kk = k % len(inside)
        m |= (ang >= bins[kk]) & (ang < bins[kk + 1])
    n = np.array([np.cos(mid), np.sin(mid)])
    off = float(np.median((P[m] - [cx, cy]) @ n))
    return off if 0 < off < r else None


def clip_hull(h: Hull, z_top: float) -> Hull:
    occ = h.occ.copy()
    zc = h.origin[2] + np.arange(occ.shape[2]) * h.voxel
    occ[:, :, zc > z_top] = False
    return Hull(occ, h.origin, h.voxel)


def _pca_extents(xy: np.ndarray, Vt: np.ndarray, c: np.ndarray, q: float = 0.5) -> np.ndarray:
    pr = (xy - c) @ Vt.T
    return np.percentile(pr, 100 - q, axis=0) - np.percentile(pr, q, axis=0)


def measure_part(images: list[np.ndarray], cal: Calibration, voxel: float = 0.3, n_boot: int = 8,
                 masks: list[np.ndarray] | None = None) -> tuple[PartMeasurement, Hull, list[np.ndarray]]:
    views = cal.views
    if masks is None:
        masks = [segment(im, v) if v is not None else np.zeros(im.shape[:2], np.uint8) for im, v in zip(images, views)]
    hull0 = visual_hull(masks, views, voxel)
    meas_notes: list[str] = []
    if len(hull0.points()) == 0:
        return PartMeasurement(0, 0, 0, 0, 0, notes=["empty hull"]), hull0, masks
    grads = _grad_images(images, views)
    z_top, _ = top_height_sweep(images, views, hull0, grads)
    if z_top < hull0.height - voxel:
        meas_notes.append(f"height from edge plane-sweep ({z_top:.2f}) < hull ({hull0.height:.2f})")
    hull = clip_hull(hull0, z_top)
    pts = hull.points()
    xy = pts[:, :2]
    c = xy.mean(0)
    U, S, Vt = np.linalg.svd(xy - c, full_matrices=False)
    P2, N2 = outline_points(hull)
    hull_ext = (_pca_extents(P2, Vt, c, 1.0) + voxel) if len(P2) > 20 else (_pca_extents(xy, Vt, c, 0.0) + voxel)
    _, offs = refine_outline(P2, N2, z_top, views, grads)
    if len(offs) > 30:
        # robust: hull extent corrected by the median sub-pixel offset of the top outline (outliers = shadows)
        ext = hull_ext + 2 * float(np.clip(np.median(offs), -1.5, 0.3))
    else:
        ext = hull_ext
        meas_notes.append("extent from hull only")
    fp_area = float(footprint_image(hull).sum() * voxel * voxel)
    circles = []
    for ci in circles_from_hull(hull):
        ci.z = min(ci.z, z_top)
        circles.append(refine_circle(ci, images, views))
    m = PartMeasurement(height=z_top, extent_major=float(ext[0]), extent_minor=float(ext[1]),
                        axis_angle_deg=float(np.degrees(np.arctan2(Vt[0, 1], Vt[0, 0]))), footprint_area=fp_area,
                        circles=circles, n_views_used=sum(v is not None for v in views), calib_rms_px=cal.rms_px,
                        notes=meas_notes)
    # bootstrap the measurements over views
    used = [i for i, v in enumerate(views) if v is not None]
    rng = np.random.default_rng(1)
    hs, e0, e1 = [], [], []
    for _ in range(n_boot if len(used) >= 4 else 0):
        sub = sorted(set(rng.choice(used, len(used), replace=True).tolist()))
        if len(sub) < 3:
            continue
        gsub = {k: g for k, g in grads.items() if k in sub}
        zb, _ = top_height_sweep(images, views, hull0, gsub)
        hs.append(zb)
        _, ob = refine_outline(P2, N2, zb, views, gsub)
        if len(ob) > 30:
            ee = hull_ext + 2 * float(np.clip(np.median(ob), -1.5, 0.3))
            e0.append(ee[0])
            e1.append(ee[1])
    floor = 0.05
    # model term: a large hull-vs-sweep gap means the top may be a small raised feature (see sweep docstring)
    gap_term = 0.25 * max(0.0, hull0.height - z_top)
    m.sigma = {"height": float(np.hypot(np.std(hs) if len(hs) > 2 else voxel, gap_term) + floor),
               "extent_major": float(np.std(e0) + floor) if len(e0) > 2 else voxel,
               "extent_minor": float(np.std(e1) + floor) if len(e1) > 2 else voxel}
    return m, hull, masks
