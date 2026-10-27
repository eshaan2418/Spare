"""Generic random compositions (feature graph -> CadQuery) for SFT diversity.

Teaches the generator to translate the structured representation into CadQuery *compositionally*
rather than memorising the 8 train-family templates. The vocabulary is generic CAD (plate, cylinder,
through/blind holes on any axis, bosses, pockets, slots, lugs, fillets, chamfers) and was fixed without
reference to test-family feature types (DECISIONS D-021). No mating/fit specs: compositions are SFT-only.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from spare.common.provenance import ROOT
from spare.common.sandbox import ProgramRunner
from spare.repr.render import render
from spare.synth.family import fmt_params, r2

FRAME = "Z up = build direction, part on z >= 0"


def _f(x):
    return r2(float(x))


def compose(rng: np.random.Generator) -> tuple[dict, str]:
    body_kind = rng.choice(["plate", "cylinder"], p=[0.65, 0.35])
    prm: list[tuple[str, float, str]] = []
    code: list[str] = []
    feats: list[dict] = []
    if body_kind == "plate":
        L, W, T = _f(rng.uniform(15, 70)), _f(rng.uniform(10, 45)), _f(rng.uniform(2, 12))
        prm += [("length", L, "X"), ("width", W, "Y"), ("thickness", T, "Z")]
        body = {"type": "plate", "length_x": L, "width_y": W, "thickness": T, "centered_xy": True}
        code.append('body = cq.Workplane("XY").box(length, width, thickness, centered=(True, True, False))')
        if rng.random() < 0.4:
            fr = _f(rng.uniform(0.5, min(L, W) / 5))
            prm.append(("corner_radius", fr, ""))
            code.append('body = body.edges("|Z").fillet(corner_radius)')
            feats.append({"id": "corners", "type": "fillet", "edges": "vertical edges", "radius": fr})
        xr, yr, top = L / 2, W / 2, T
    else:
        D, H = _f(rng.uniform(10, 50)), _f(rng.uniform(3, 25))
        prm += [("diameter", D, ""), ("height", H, "")]
        body = {"type": "cylinder", "axis": "z", "diameter": D, "height": H, "center_xy": [0, 0]}
        code.append('body = cq.Workplane("XY").circle(diameter / 2).extrude(height)')
        xr = yr = D / 2 / 1.5
        top = H
    n_feat = int(rng.integers(1, 5))
    used = []
    for k in range(n_feat):
        kind = rng.choice(["through_holes", "blind_hole", "boss", "pocket", "slot", "lug", "cross_hole", "chamfer"])
        i = k + 1
        if kind == "through_holes":
            n = int(rng.integers(1, 5))
            d = _f(rng.uniform(1.5, min(8, xr, yr)))
            pts = [[_f(rng.uniform(-xr + d, xr - d)), _f(rng.uniform(-yr + d, yr - d))] for _ in range(n)]
            prm.append((f"hole{i}_diameter", d, ""))
            code.append(f"hole{i}_centers = {pts}")
            code.append(f'body = body.cut(cq.Workplane("XY").workplane(offset=-1).pushPoints(hole{i}_centers)'
                        f'.circle(hole{i}_diameter / 2).extrude({_f(top + 2)}))')
            feats.append({"id": f"hole{i}", "type": "through_holes", "axis": "z", "centers": pts, "diameter": d})
        elif kind == "blind_hole":
            d = _f(rng.uniform(2, min(10, xr, yr)))
            dep = _f(rng.uniform(0.3, 0.8) * top)
            c = [_f(rng.uniform(-xr + d, xr - d)), _f(rng.uniform(-yr + d, yr - d))]
            prm += [(f"bore{i}_diameter", d, ""), (f"bore{i}_depth", dep, "from the top face")]
            code.append(f'body = body.cut(cq.Workplane("XY").workplane(offset={_f(top)} - bore{i}_depth)'
                        f'.center({c[0]}, {c[1]}).circle(bore{i}_diameter / 2).extrude(bore{i}_depth + 1))')
            feats.append({"id": f"bore{i}", "type": "blind_hole", "axis": "z", "at": c, "diameter": d,
                          "depth": dep, "from": "top face"})
        elif kind == "boss":
            d = _f(rng.uniform(3, min(14, 2 * xr, 2 * yr)))
            h = _f(rng.uniform(1.5, 10))
            c = [_f(rng.uniform(-xr + d / 2, xr - d / 2)), _f(rng.uniform(-yr + d / 2, yr - d / 2))]
            prm += [(f"boss{i}_diameter", d, ""), (f"boss{i}_height", h, "above the top face")]
            code.append(f'body = body.union(cq.Workplane("XY").workplane(offset={_f(top)}).center({c[0]}, {c[1]})'
                        f'.circle(boss{i}_diameter / 2).extrude(boss{i}_height))')
            feats.append({"id": f"boss{i}", "type": "boss", "axis": "z", "at": c, "diameter": d, "height": h,
                          "on": "top face"})
        elif kind == "pocket" and body_kind == "plate":
            px, py = _f(rng.uniform(0.2, 0.6) * 2 * xr), _f(rng.uniform(0.2, 0.6) * 2 * yr)
            dep = _f(rng.uniform(0.2, 0.6) * top)
            prm += [(f"pocket{i}_x", px, ""), (f"pocket{i}_y", py, ""), (f"pocket{i}_depth", dep, "")]
            code.append(f'body = body.cut(cq.Workplane("XY").workplane(offset={_f(top)} - pocket{i}_depth)'
                        f'.rect(pocket{i}_x, pocket{i}_y).extrude(pocket{i}_depth + 1))')
            feats.append({"id": f"pocket{i}", "type": "rect_pocket", "at": [0, 0], "size_xy": [px, py],
                          "depth": dep, "from": "top face"})
        elif kind == "slot" and body_kind == "plate":
            sl, sw = _f(rng.uniform(0.3, 0.7) * 2 * xr), _f(rng.uniform(1.5, min(6, yr)))
            prm += [(f"slot{i}_length", sl, "end to end"), (f"slot{i}_width", sw, "")]
            code.append(f'body = body.cut(cq.Workplane("XY").workplane(offset=-1).slot2D(slot{i}_length, slot{i}_width)'
                        f'.extrude({_f(top + 2)}))')
            feats.append({"id": f"slot{i}", "type": "through_slot", "axis": "z", "at": [0, 0], "direction": "x",
                          "length": sl, "width": sw})
        elif kind == "lug" and body_kind == "plate":
            r = _f(rng.uniform(1.5, min(6, top * 1.5)))
            ax = str(rng.choice(["x", "y"]))
            span = _f((2 * xr if ax == "x" else 2 * yr) * rng.uniform(0.3, 1.0))
            off = _f(rng.uniform(-0.5, 0.5) * (2 * yr if ax == "x" else 2 * xr) * 0.5)
            prm += [(f"lug{i}_radius", r, ""), (f"lug{i}_length", span, "")]
            v = "(1, 0, 0)" if ax == "x" else "(0, 1, 0)"
            p0 = f"(-lug{i}_length / 2, {off}, {_f(top)})" if ax == "x" else f"({off}, -lug{i}_length / 2, {_f(top)})"
            code.append(f'body = body.union(cq.Workplane("XY").add(cq.Solid.makeCylinder(lug{i}_radius, lug{i}_length, '
                        f'cq.Vector{p0}, cq.Vector{v})))')
            feats.append({"id": f"lug{i}", "type": "cylinder", "axis": ax, "center": [off, _f(top)] if ax == "x"
                          else [off, _f(top)], "center_meaning": "(y, z)" if ax == "x" else "(x, z)", "radius": r,
                          "length": span, "centered_on": "body centre"})
        elif kind == "cross_hole" and body_kind == "plate" and top >= 4:
            d = _f(rng.uniform(1.5, top * 0.6))
            ax = str(rng.choice(["x", "y"]))
            z = _f(top / 2)
            off = _f(rng.uniform(-0.6, 0.6) * (yr if ax == "x" else xr))
            prm.append((f"cross{i}_diameter", d, ""))
            ln = _f((2 * xr if ax == "x" else 2 * yr) + 2)
            v = "(1, 0, 0)" if ax == "x" else "(0, 1, 0)"
            p0 = f"({-ln / 2}, {off}, {z})" if ax == "x" else f"({off}, {-ln / 2}, {z})"
            code.append(f'body = body.cut(cq.Workplane("XY").add(cq.Solid.makeCylinder(cross{i}_diameter / 2, {ln}, '
                        f'cq.Vector{p0}, cq.Vector{v})))')
            feats.append({"id": f"cross{i}", "type": "through_hole", "axis": ax,
                          ("center_yz" if ax == "x" else "center_xz"): [off, z], "diameter": d})
        elif kind == "chamfer" and "chamfer" not in used:
            c = _f(rng.uniform(0.3, 1.0))
            prm.append(("top_chamfer", c, ""))
            code.append('body = body.faces(">Z").edges().chamfer(top_chamfer)' if body_kind == "cylinder" and not feats
                        else "# (chamfer skipped: top face already modified)")
            if body_kind == "cylinder" and not feats:
                feats.append({"id": "top_edge", "type": "chamfer", "edges": "top circular edge", "size": c})
            used.append("chamfer")
    code.append("result = body")
    graph = {"part_type": f"generic {body_kind} part", "frame": FRAME, "body": body, "features": feats}
    src = "import cadquery as cq\n\n# Generic part\n# Units: mm. Frame: " + FRAME + "\n" + fmt_params(prm) + "\n" + \
        "\n".join(code) + "\n"
    return graph, src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    items = []
    for i in range(a.n):
        g, src = compose(rng)
        mat = "PETG" if rng.random() < 0.3 else "PLA"
        items.append({"uid": f"compose_{i:05d}", "family": "compose_simple" if len(g["features"]) <= 1 else
                      "compose", "repr": render(g, mat, "generic_petg_0.4" if mat == "PETG" else "generic_pla_0.4"),
                      "program": src})
    with ProgramRunner(timeout=30) as run:
        res = run.run_many([it["program"] for it in items], trusted=True)
    keep = [it for it, r in zip(items, res) if r.ok]
    out = ROOT / "data" / "synthetic" / "compose.jsonl"
    with open(out, "w") as f:
        for it in keep:
            f.write(json.dumps(it) + "\n")
    print({"generated": len(items), "kept": len(keep)})


if __name__ == "__main__":
    main()
