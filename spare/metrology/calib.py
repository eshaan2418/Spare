"""Card detection, camera calibration and per-view pose (known metric scale from the card).

World frame = card frame (see spare/synth/render.py): board (x, y_b) -> world (x, -y_b, 0).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from spare.capture.card import CARD


@dataclass
class ViewDetection:
    img_pts: np.ndarray  # (N, 2)
    obj_pts: np.ndarray  # (N, 3) world
    n_markers: int
    n_charuco: int


@dataclass
class CalibratedView:
    K: np.ndarray
    dist: np.ndarray
    R: np.ndarray
    t: np.ndarray
    reproj_px: float
    size: tuple[int, int]

    def project(self, X: np.ndarray) -> np.ndarray:
        rv, _ = cv2.Rodrigues(self.R)
        uv, _ = cv2.projectPoints(np.asarray(X, np.float64).reshape(-1, 1, 3), rv, self.t, self.K, self.dist)
        return uv.reshape(-1, 2)

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t


@dataclass
class Calibration:
    views: list[CalibratedView | None]
    rms_px: float
    used: list[int] = field(default_factory=list)
    method: str = "charuco+markers"


def _board_to_world(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, np.float64).reshape(-1, 3)
    return np.c_[p[:, 0], -p[:, 1], np.zeros(len(p))]


def detect(img: np.ndarray) -> ViewDetection | None:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    det = CARD.detector()
    ch_corners, ch_ids, m_corners, m_ids = det.detectBoard(gray)
    if m_ids is None or len(m_ids) < 4:
        return None
    board = CARD.board()
    obj_all = board.getObjPoints()
    ids_all = list(np.asarray(board.getIds()).ravel())
    ip, op = [], []
    for c, i in zip(m_corners, np.asarray(m_ids).ravel()):
        if int(i) in ids_all:
            k = ids_all.index(int(i))
            ip.append(np.asarray(c).reshape(4, 2))
            op.append(_board_to_world(obj_all[k]))
    nch = 0
    if ch_ids is not None and len(ch_ids):
        cc = board.getChessboardCorners()
        for c, i in zip(np.asarray(ch_corners).reshape(-1, 2), np.asarray(ch_ids).ravel()):
            ip.append(c[None])
            op.append(_board_to_world(cc[int(i)][None]))
        nch = len(ch_ids)
    return ViewDetection(np.vstack(ip).astype(np.float32), np.vstack(op).astype(np.float32), len(m_ids), nch)


def calibrate(images: list[np.ndarray], fix_k3: bool = True, focal_guess: float | None = None) -> Calibration:
    """Joint intrinsics (shared across views, same phone camera) + per-view extrinsics."""
    dets = [detect(im) for im in images]
    h, w = images[0].shape[:2]
    used = [i for i, d in enumerate(dets) if d is not None and len(d.img_pts) >= 12]
    if len(used) == 0:
        return Calibration([None] * len(images), float("inf"), [])
    f0 = focal_guess or 0.9 * w
    K0 = np.array([[f0, 0, w / 2], [0, f0, h / 2], [0, 0, 1]], np.float64)
    flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_ZERO_TANGENT_DIST | (cv2.CALIB_FIX_K3 if fix_k3 else 0)
    if len(used) < 3:  # too few views for intrinsics: fix principal point and aspect
        flags |= cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2
    obj = [dets[i].obj_pts for i in used]  # type: ignore[union-attr]
    ipt = [dets[i].img_pts for i in used]  # type: ignore[union-attr]
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj, ipt, (w, h), K0, np.zeros(5), flags=flags)
    views: list[CalibratedView | None] = [None] * len(images)
    for j, i in enumerate(used):
        R, _ = cv2.Rodrigues(rvecs[j])
        proj, _ = cv2.projectPoints(obj[j].reshape(-1, 1, 3).astype(np.float64), rvecs[j], tvecs[j], K, dist)
        err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - ipt[j]) ** 2, 1))))
        views[i] = CalibratedView(K, dist, R, tvecs[j].ravel(), err, (w, h))
    return Calibration(views, float(rms), used)
