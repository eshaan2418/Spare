"""Photorealistic-ish synthetic photos of a part on the virtual scale card.

World (card) frame: X right along the board, Y up the page (= -board y), Z up out
of the paper. Board object points (x, y_b, 0) map to world (x, -y_b, 0).
Camera: OpenCV pinhole, x_cam = R @ X_world + t.

Rendering: textured card by homography; part by painter's-algorithm triangle
rasterisation at 2x supersampling with Lambert + Blinn-Phong shading; cast
shadow on the card; sensor noise, blur, exposure, JPEG.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

from spare.capture.card import CARD, render_page

PAGE_PPM = 10.0  # page texture px/mm


@lru_cache(maxsize=1)
def page_texture() -> np.ndarray:
    return render_page(CARD, dpi=int(round(PAGE_PPM * 25.4)))


def stage_center_world() -> np.ndarray:
    x0, y0, x1, y1 = CARD.stage_mm
    return np.array([(x0 + x1) / 2, -(y0 + y1) / 2, 0.0])


@dataclass
class Camera:
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    size: tuple[int, int]  # (w, h)

    def project(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Xc = X @ self.R.T + self.t
        uv = Xc @ self.K.T
        return uv[:, :2] / uv[:, 2:3], Xc[:, 2]

    @property
    def center_world(self) -> np.ndarray:
        return -self.R.T @ self.t


def look_at(eye: np.ndarray, target: np.ndarray, up_hint: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    if up_hint is None:
        up_hint = np.array([0, 1.0, 0])
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up_hint)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(z, np.array([1.0, 0, 0]))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])  # rows: camera axes in world
    t = -R @ eye
    return R, t


def sample_intrinsics(rng: np.random.Generator, size=(1600, 1200)) -> np.ndarray:
    w, h = size
    f = rng.uniform(0.84, 1.0) * w  # ~ phone main camera (26 mm equiv.) FOV
    return np.array([[f, 0, w / 2 + rng.normal(0, 4)], [0, f, h / 2 + rng.normal(0, 4)], [0, 0, 1.0]])


def sample_camera(rng: np.random.Generator, kind: str, size=(1600, 1200), target=None, dist=None,
                  K: np.ndarray | None = None) -> Camera:
    K = sample_intrinsics(rng, size) if K is None else K
    tgt = (stage_center_world() if target is None else np.asarray(target, float)) + np.r_[rng.normal(0, 4, 2), 0]
    if kind == "top":
        elev = np.radians(rng.uniform(80, 89))
    elif kind == "low":
        elev = np.radians(rng.uniform(25, 35))
    else:
        elev = np.radians(rng.uniform(40, 65))
    az = rng.uniform(0, 2 * np.pi)
    # guide: the grey stage fills most of the frame with >= 2 marker rows visible
    d = dist if dist is not None else rng.uniform(170, 230) * (1.15 if kind == "low" else 1.0)
    eye = tgt + d * np.array([np.cos(elev) * np.cos(az), np.cos(elev) * np.sin(az), np.sin(elev)])
    R, t = look_at(eye, tgt, up_hint=np.array([np.cos(az + np.pi / 2), np.sin(az + np.pi / 2), 0.0]))
    return Camera(K, R, t, size)


def place_on_card(vertices: np.ndarray, yaw: float, center_xy: np.ndarray) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    v = vertices @ Rz.T
    v[:, :2] += center_xy - 0.5 * (v[:, :2].min(0) + v[:, :2].max(0))
    v[:, 2] -= v[:, 2].min()
    return v


def _card_homography(cam: Camera, ss: float) -> np.ndarray:
    ox, oy = CARD.origin_mm
    A = np.array([[1.0, 0, -ox], [0, -1.0, oy], [0, 0, 1.0]])  # page mm -> world XY
    S = np.diag([1 / PAGE_PPM, 1 / PAGE_PPM, 1.0])
    P = cam.K @ np.c_[cam.R[:, 0], cam.R[:, 1], cam.t]
    Hs = np.diag([ss, ss, 1.0])
    return Hs @ P @ A @ S


def render(cam: Camera, verts: np.ndarray | None, faces: np.ndarray | None, rng: np.random.Generator,
           color=None, ss: int = 2, extra_meshes: list | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (BGR uint8 image, part mask uint8) at cam.size."""
    w, h = cam.size
    W, H = w * ss, h * ss
    # background table texture
    base = rng.uniform(60, 200, 3)
    bg = np.ones((H, W, 3), np.float32) * base
    noise = cv2.resize(rng.normal(0, 12, (H // 32 + 1, W // 32 + 1, 1)).astype(np.float32), (W, H))[..., None]
    bg += noise
    page = page_texture().astype(np.float32)
    Hm = _card_homography(cam, ss)
    card = cv2.warpPerspective(page, Hm, (W, H), flags=cv2.INTER_LINEAR, borderValue=-1)
    card_mask = card >= 0
    paper_tint = rng.uniform(0.92, 1.0, 3)
    img = bg.copy()
    img[card_mask] = card[card_mask, None] * paper_tint[None, :]
    # light
    l_az, l_el = rng.uniform(0, 2 * np.pi), np.radians(rng.uniform(35, 75))
    L = -np.array([np.cos(l_el) * np.cos(l_az), np.cos(l_el) * np.sin(l_az), np.sin(l_el)])  # direction of travel
    meshes = []
    if verts is not None:
        meshes.append((verts, faces, color, True))
    for m in extra_meshes or []:
        meshes.append((m[0], m[1], m[2], False))
    mask = np.zeros((H, W), np.uint8)
    # shadows
    shadow = np.zeros((H, W), np.float32)
    for v, f, _, _ in meshes:
        vs = v - (v[:, 2:3] / L[2]) * L[None, :]
        vs[:, 2] = 0
        uv, _ = cam.project(vs)
        tri = (uv[f] * ss).astype(np.int32)
        for t in tri:
            cv2.fillConvexPoly(shadow, t, 1.0)
    shadow = cv2.GaussianBlur(shadow, (0, 0), 3 * ss)
    img = img * (1 - rng.uniform(0.25, 0.45) * shadow)[..., None]
    # part rasterisation (painter's algorithm, back-face culled)
    eye = cam.center_world
    for v, f, col, is_part in meshes:
        col = np.asarray(col if col is not None else rng.uniform(30, 230, 3), np.float32)
        tri_w = v[f]
        n = np.cross(tri_w[:, 1] - tri_w[:, 0], tri_w[:, 2] - tri_w[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        cen = tri_w.mean(1)
        view = eye[None] - cen
        view /= np.linalg.norm(view, axis=1, keepdims=True)
        front = (n * view).sum(1) > 0
        uv, z = cam.project(v)
        zc = z[f].mean(1)
        order = np.argsort(-zc)
        diff = np.clip((n * -L[None]).sum(1), 0, 1)
        half = (-L[None] + view)
        half /= np.linalg.norm(half, axis=1, keepdims=True)
        spec = np.clip((n * half).sum(1), 0, 1) ** 40
        shade = 0.35 + 0.65 * diff
        pix = (uv[f] * ss).astype(np.int32)
        for k in order:
            if not front[k]:
                continue
            c = col * shade[k] + 255 * 0.25 * spec[k]
            cv2.fillConvexPoly(img, pix[k], c.tolist())
            if is_part:
                cv2.fillConvexPoly(mask, pix[k], 255)
            else:
                cv2.fillConvexPoly(mask, pix[k], 0)
    # exposure / vignetting
    yy, xx = np.mgrid[0:32, 0:32].astype(np.float32) / 31.0
    r2 = cv2.resize(((xx - 0.5) ** 2 + (yy - 0.5) ** 2) * 4, (W, H))
    img = img * (rng.uniform(0.85, 1.1) * (1 - rng.uniform(0.05, 0.2) * r2))[..., None]
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_AREA)
    blur = rng.uniform(0.3, 1.0)
    img = cv2.GaussianBlur(img, (0, 0), blur)
    img += rng.normal(0, rng.uniform(1.5, 4.0), img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    q = int(rng.integers(80, 96))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img, (mask > 127).astype(np.uint8) * 255


VIEW_PLAN = ["top", "top", "oblique", "oblique", "oblique", "oblique", "low", "low"]


def render_part_views(mesh, rng: np.random.Generator, n_views: int = 8, size=(1600, 1200),
                      yaw: float | None = None) -> dict:
    """Render a capture set; returns images, masks, cameras and the placed vertices (ground truth)."""
    yaw = float(rng.uniform(0, 2 * np.pi)) if yaw is None else yaw
    v = place_on_card(np.asarray(mesh.vertices, float), yaw, stage_center_world()[:2] + rng.normal(0, 5, 2))
    f = np.asarray(mesh.faces)
    col = rng.uniform(20, 235, 3)
    out: dict = {"images": [], "masks": [], "cameras": [], "verts": v, "faces": f, "yaw": yaw, "color": col}
    plan = (VIEW_PLAN * 3)[:n_views]
    K = sample_intrinsics(rng, size)
    for k in plan:
        cam = sample_camera(rng, k, size, target=np.r_[v[:, :2].mean(0), 0.0], K=K)
        im, m = render(cam, v, f, rng, col)
        out["images"].append(im)
        out["masks"].append(m)
        out["cameras"].append(cam)
    return out
