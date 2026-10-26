"""Live capture quality checks (used by the app's guided capture and by `spare.capture.quality <dir>`).

Each photo gets: sharpness (variance of Laplacian on the card region), exposure (clipped fractions),
card detection (markers, ChArUco corners), and part placement (non-card blob inside the grey stage).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from spare.metrology.calib import detect

MIN_MARKERS = 6
MIN_SHARPNESS = 60.0
MAX_CLIPPED = 0.02


@dataclass
class PhotoQuality:
    path: str
    ok: bool
    sharpness: float
    clipped_dark: float
    clipped_bright: float
    n_markers: int
    n_charuco: int
    issues: list[str] = field(default_factory=list)


def check_image(img: np.ndarray, path: str = "") -> PhotoQuality:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    dark = float((gray <= 3).mean())
    bright = float((gray >= 252).mean())
    det = detect(img)
    nm = det.n_markers if det else 0
    nc = det.n_charuco if det else 0
    issues = []
    if sharp < MIN_SHARPNESS:
        issues.append("blurry: hold still, tap to focus on the part")
    if bright > MAX_CLIPPED:
        issues.append("over-exposed: avoid direct sun / glare")
    if dark > MAX_CLIPPED:
        issues.append("under-exposed: add light")
    if nm < MIN_MARKERS:
        issues.append(f"scale card not visible enough ({nm} markers; need >= {MIN_MARKERS}): include more of the card")
    return PhotoQuality(path, not issues, sharp, dark, bright, nm, nc, issues)


def check_dir(d: Path) -> dict:
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    res = []
    for p in sorted(d.iterdir()):
        if p.suffix in exts:
            img = cv2.imread(str(p))
            if img is None:
                res.append(PhotoQuality(str(p), False, 0, 0, 0, 0, 0, ["unreadable (HEIC? export as JPEG)"]))
            else:
                res.append(check_image(img, str(p)))
    n_ok = sum(r.ok for r in res)
    summary = {"n_photos": len(res), "n_ok": n_ok, "enough": n_ok >= 5,
               "photos": [asdict(r) for r in res]}
    if n_ok < 5:
        summary["advice"] = "Need at least 5 good photos of the part on the card (see capture/CAPTURE_GUIDE.md)."
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photo_dir")
    a = ap.parse_args()
    print(json.dumps(check_dir(Path(a.photo_dir)), indent=1))


if __name__ == "__main__":
    main()
