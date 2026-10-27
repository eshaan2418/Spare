"""Completion of broken/missing regions.

* Rotational symmetry: a round part with a missing wedge/chunk still shows an arc of its outer contour;
  a RANSAC circle on the footprint contour recovers the intact diameter and centre (the naive extent
  under-estimates or the circle detector rejects the contour).
* Mirror symmetry: footprint reflected about the principal axis, used when one half is missing.
* Part-family priors: for features that are entirely missing (a sheared pin, a snapped hook), the
  generator's template supplies the feature and its size comes from the measured remaining features
  plus the type's design rules (tolerance engine / DfAM); see spare/features/infer.py.
"""

from __future__ import annotations

import cv2
import numpy as np

from spare.metrology.reconstruct import Hull, footprint_image


def _circle3(p):
    (x1, y1), (x2, y2), (x3, y3) = p
    a = np.array([[x2 - x1, y2 - y1], [x3 - x1, y3 - y1]], float)
    b = 0.5 * np.array([x2 ** 2 - x1 ** 2 + y2 ** 2 - y1 ** 2, x3 ** 2 - x1 ** 2 + y3 ** 2 - y1 ** 2])
    if abs(np.linalg.det(a)) < 1e-9:
        return None
    c = np.linalg.solve(a, b)
    return c[0], c[1], float(np.hypot(x1 - c[0], y1 - c[1]))


def ransac_circle(P: np.ndarray, thr: float = 0.35, iters: int = 400, seed: int = 0, min_frac: float = 0.3):
    """Robust circle (cx, cy, r, inlier_frac, arc_coverage_deg) on 2D points; None if no support."""
    if len(P) < 10:
        return None
    rng = np.random.default_rng(seed)
    best, bi = None, None
    for _ in range(iters):
        c = _circle3(P[rng.choice(len(P), 3, replace=False)])
        if c is None or c[2] > 200:
            continue
        inl = np.abs(np.hypot(P[:, 0] - c[0], P[:, 1] - c[1]) - c[2]) < thr
        if bi is None or inl.sum() > bi.sum():
            best, bi = c, inl
    if best is None or bi is None or bi.mean() < min_frac:
        return None
    Q = P[bi]
    A = np.c_[2 * Q, np.ones(len(Q))]
    sol, *_ = np.linalg.lstsq(A, (Q ** 2).sum(1), rcond=None)
    cx, cy = sol[0], sol[1]
    r = float(np.sqrt(max(sol[2] + cx * cx + cy * cy, 1e-9)))
    ang = np.degrees(np.arctan2(Q[:, 1] - cy, Q[:, 0] - cx))
    cov = len(np.unique(np.floor((ang + 180) / 10))) * 10.0
    return float(cx), float(cy), r, float(bi.mean()), cov


def outer_contour(h: Hull) -> np.ndarray:
    fp = footprint_image(h)
    cnts, _ = cv2.findContours(fp.T.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return np.zeros((0, 2))
    c = max(cnts, key=len)
    return h.origin[:2] + c.reshape(-1, 2).astype(float) * h.voxel


def complete_round(h: Hull, offset: float = 0.0) -> dict | None:
    """Intact outer diameter of a (possibly broken) round part from the arc of its outer contour.
    ``offset`` is the per-side sub-pixel correction from metrology (negative = hull inflated)."""
    P = outer_contour(h)
    c = ransac_circle(P)
    if c is None:
        return None
    cx, cy, r, frac, cov = c
    return {"center": [cx, cy], "diameter": 2 * (r + 0.5 * h.voxel + offset), "inlier_frac": frac,
            "arc_coverage_deg": cov, "symmetry": "rotational"}


def mirror_complete(h: Hull) -> dict:
    """Extents after reflecting the footprint about its principal axis (one half missing)."""
    P = outer_contour(h)
    c = P.mean(0)
    _, _, Vt = np.linalg.svd(P - c, full_matrices=False)
    pr = (P - c) @ Vt.T
    half = np.abs(pr).max(0)
    return {"extent_major": float(2 * half[0]), "extent_minor": float(2 * half[1]), "symmetry": "mirror"}
