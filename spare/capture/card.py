"""Printable ChArUco scale card (A4) with a 100 mm sanity ruler.

Card frame: origin at the board's top-left chessboard corner, X to the right,
Y down the page (OpenCV board convention), Z out of the paper (toward camera)
-- we flip to Z up = away from the card into the scene, so part points have z>=0.
All units mm.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

DPI = 600
A4_MM = (210.0, 297.0)


@dataclass(frozen=True)
class CardSpec:
    cols: int = 7
    rows: int = 9
    square_mm: float = 22.0
    marker_mm: float = 16.5
    dictionary: int = cv2.aruco.DICT_5X5_250
    origin_mm: tuple[float, float] = (28.0, 22.0)  # board top-left on the page
    ruler_y_mm: float = 248.0
    version: str = "spare-card-v1"
    stage_squares: tuple[int, int, int, int] = (1, 2, 6, 7)  # (col0, row0, col1, row1) exclusive end: grey stage
    stage_gray: int = 150

    @property
    def stage_mm(self) -> tuple[float, float, float, float]:
        c0, r0, c1, r1 = self.stage_squares
        s = self.square_mm
        return c0 * s, r0 * s, c1 * s, r1 * s

    @property
    def board_mm(self) -> tuple[float, float]:
        return self.cols * self.square_mm, self.rows * self.square_mm

    def board(self):
        d = cv2.aruco.getPredefinedDictionary(self.dictionary)
        b = cv2.aruco.CharucoBoard((self.cols, self.rows), self.square_mm, self.marker_mm, d)
        return b

    def detector(self):
        params = cv2.aruco.CharucoParameters()
        dp = cv2.aruco.DetectorParameters()
        dp.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        return cv2.aruco.CharucoDetector(self.board(), params, dp)


CARD = CardSpec()


def mm2px(mm: float, dpi: int = DPI) -> int:
    return int(round(mm / 25.4 * dpi))


def render_board_image(spec: CardSpec = CARD, px_per_mm: float = 10.0) -> np.ndarray:
    """Board only (white margin-free), at px_per_mm resolution; pixel (u,v) <-> mm (u/px_per_mm, v/px_per_mm)."""
    w, h = spec.board_mm
    img = spec.board().generateImage((int(round(w * px_per_mm)), int(round(h * px_per_mm))), marginSize=0, borderBits=1)
    x0, y0, x1, y1 = (int(round(v * px_per_mm)) for v in spec.stage_mm)
    img[y0:y1, x0:x1] = spec.stage_gray
    return img


def render_page(spec: CardSpec = CARD, dpi: int = DPI) -> np.ndarray:
    W, H = mm2px(A4_MM[0], dpi), mm2px(A4_MM[1], dpi)
    page = np.full((H, W), 255, np.uint8)
    ppm = dpi / 25.4
    board = render_board_image(spec, ppm)
    x0, y0 = mm2px(spec.origin_mm[0], dpi), mm2px(spec.origin_mm[1], dpi)
    page[y0:y0 + board.shape[0], x0:x0 + board.shape[1]] = board
    # 100 mm ruler with 1 mm ticks
    rx0 = mm2px(55.0, dpi)
    ry = mm2px(spec.ruler_y_mm, dpi)
    cv2.line(page, (rx0, ry), (rx0 + mm2px(100.0, dpi), ry), 0, max(1, mm2px(0.25, dpi)))
    for i in range(101):
        L = 6.0 if i % 10 == 0 else (4.0 if i % 5 == 0 else 2.5)
        x = rx0 + int(round(i * ppm))
        cv2.line(page, (x, ry), (x, ry - mm2px(L, dpi)), 0, max(1, mm2px(0.18, dpi)))
        if i % 10 == 0:
            cv2.putText(page, str(i), (x - mm2px(1.5, dpi), ry + mm2px(5, dpi)), cv2.FONT_HERSHEY_SIMPLEX,
                        dpi / 600 * 1.4, 0, max(1, mm2px(0.25, dpi)))
    txt = [f"{spec.version}  ChArUco {spec.cols}x{spec.rows}  square {spec.square_mm:.2f} mm  marker {spec.marker_mm:.2f} mm  DICT_5X5_250",
           "PRINT AT 100% (actual size). Check: ruler 0-100 must measure 100.0 mm with calipers.",
           "Place the part on the GREY STAGE; keep the card flat; keep at least 2 marker rows visible in every photo."]
    for i, t in enumerate(txt):
        cv2.putText(page, t, (mm2px(15, dpi), mm2px(266 + 7 * i, dpi)), cv2.FONT_HERSHEY_SIMPLEX, dpi / 600 * 1.05, 0,
                    max(1, mm2px(0.2, dpi)))
    return page


def save_card(out_dir: Path, spec: CardSpec = CARD) -> dict:
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    page = render_page(spec)
    png = out_dir / "scale_card_A4.png"
    pdf = out_dir / "scale_card_A4.pdf"
    cv2.imwrite(str(png), page)
    Image.fromarray(page).convert("L").save(pdf, resolution=DPI)
    return {"png": str(png), "pdf": str(pdf)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent))
    print(save_card(Path(ap.parse_args().out)))


if __name__ == "__main__":
    main()
