"""Prismatic / snap families: battery cover, L-bracket, cable clip, shelf peg, wall hook,
hinge leaf, rack clip, rectangular tube end cap."""

from __future__ import annotations

import math

import numpy as np

from spare.features.tolerance import FIT_BANDS, MATERIALS, printed_deviation
from spare.synth.family import (
    Family,
    FitSpec,
    LoadCase,
    Probe,
    StrengthCheck,
    cyl_points,
    fmt_params,
    header,
    plane_points,
    r2,
)

FRAME_PROFILE = "profile drawn in the XY plane on the bed and extruded along +Z (build direction)"
FRAME_FLAT = "Z up = build direction, part lying on z=0"
SAFETY_FACTOR = 2.0


def snap_strain(t: float, y: float, L: float) -> float:
    """Max bending strain of a constant-section rectangular cantilever deflected by y at its tip
    (classic snap-fit design formula: eps = 1.5 t y / L^2)."""
    return 1.5 * t * y / max(L, 1e-6) ** 2


def cantilever_stress(F: float, L: float, b: float, t: float) -> float:
    return 6.0 * F * L / max(b * t * t, 1e-9)


def _cyl(r, h, pnt, d):
    return (
        f"cq.Workplane(\"XY\").add(cq.Solid.makeCylinder({r}, {h}, cq.Vector{tuple(map(lambda v: round(v, 3), pnt))}, "
        f"cq.Vector{tuple(d)}))"
    )


# --------------------------------------------------------------------------------------
class BatteryCover(Family):
    name = "battery_cover"
    split = "train"
    part_type = "battery compartment cover with snap hook and tongue"
    n_styles = 2

    def sample_mating(self, rng):
        t = r2(rng.uniform(1.2, 2.0))
        wz = r2(t + rng.uniform(8.0, 14.0))
        return {
            "opening_length": r2(rng.uniform(30, 70)),
            "opening_width": r2(rng.uniform(20, 48)),
            "wall": r2(rng.uniform(1.5, 2.5)),
            "recess_depth": t,
            "window_z": wz,
            "window_width": r2(rng.uniform(7, 14)),
            "window_height": r2(rng.uniform(2.5, 4.5)),
            "slot_top": r2(t + rng.uniform(0.15, 0.35)),
            "slot_width": r2(rng.uniform(8, 14)),
        }

    def sample_original(self, rng, m):
        c0 = rng.uniform(0.08, 0.2)
        lip_h = r2(m["window_height"] - rng.uniform(0.8, 1.5))
        return {
            "plate_length": r2(m["opening_length"] - 2 * c0),
            "plate_width": r2(m["opening_width"] - 2 * c0),
            "plate_thickness": m["recess_depth"],
            "corner_radius": r2(rng.uniform(0.5, 3.0)),
            "hook_width": r2(m["window_width"] - rng.uniform(1.0, 2.5)),
            "hook_thickness": r2(rng.uniform(1.0, 1.8)),
            "hook_setback": r2(rng.uniform(0.3, 1.0)),
            "lip_bottom_z": r2(m["window_z"] + rng.uniform(0.0, 0.08)),
            "lip_height": lip_h,
            "lip_reach": r2(c0 + rng.uniform(0.7, 1.3)),  # tip beyond the plate edge
            "tongue_length": r2(rng.uniform(1.8, 3.0)),
            "tongue_width": r2(m["slot_width"] - rng.uniform(1.0, 2.5)),
            "tongue_thickness": r2(min(rng.uniform(0.8, 1.2), m["recess_depth"] - 0.2)),
        }

    def derive_design(self, m, o, pr, material="PLA", bias=None):
        d = dict(o)
        a = self.allow("slide", "outer", 30.0, pr, bias, "plate_length")
        d["plate_length"] = r2(m["opening_length"] - 2 * a)
        d["plate_width"] = r2(m["opening_width"] - 2 * a)
        d["plate_thickness"] = m["recess_depth"]
        az = self.allow("seat", "z", 10.0, pr, bias, "lip_bottom_z")
        d["lip_bottom_z"] = r2(m["window_z"] + az)
        overlap = o["lip_reach"] - (m["opening_length"] - o["plate_length"]) / 2
        overlap = float(np.clip(overlap, 0.7, 1.4))
        # strength-aware adjustment (M6 rule): keep snap strain under the material allowable
        L = d["lip_bottom_z"] - d["plate_thickness"]
        allow_e = MATERIALS[material].allow_strain_snap * 0.9
        t = d["hook_thickness"]
        while snap_strain(t, overlap, L) > allow_e and t > 0.9:
            t -= 0.05
        while snap_strain(t, overlap, L) > allow_e and overlap > 0.65:
            overlap -= 0.05
        d["hook_thickness"] = r2(t)
        a_gap = (m["opening_length"] - d["plate_length"]) / 2
        d["lip_reach"] = r2(a_gap + overlap)
        return d

    def program(self, p, style=0):
        s = header("Battery cover: plate, snap hook at +X end, tongue at -X end (outer face on the bed)", FRAME_FLAT)
        s += fmt_params([
            ("plate_length", p["plate_length"], "critical: slide fit in opening"),
            ("plate_width", p["plate_width"], "critical: slide fit in opening"),
            ("plate_thickness", p["plate_thickness"], "critical: flush with housing"),
            ("corner_radius", p["corner_radius"], ""),
            ("hook_width", p["hook_width"], ""),
            ("hook_thickness", p["hook_thickness"], "snap beam thickness (strength)"),
            ("hook_setback", p["hook_setback"], "beam outer face inset from plate edge"),
            ("lip_bottom_z", p["lip_bottom_z"], "critical: catch face height"),
            ("lip_height", p["lip_height"], ""),
            ("lip_reach", p["lip_reach"], "critical: lip tip beyond plate edge"),
            ("tongue_length", p["tongue_length"], ""),
            ("tongue_width", p["tongue_width"], ""),
            ("tongue_thickness", p["tongue_thickness"], ""),
        ]) + "\n"
        s += (
            "plate = (\n    cq.Workplane(\"XY\")\n    .rect(plate_length, plate_width)\n    .extrude(plate_thickness)\n"
            "    .edges(\"|Z\").fillet(corner_radius)\n)\n"
            "beam_x = plate_length / 2 - hook_setback  # outer face of the snap beam\n"
            "beam_top = lip_bottom_z + lip_height\n"
        )
        if style == 0:
            s += (
                "beam = cq.Workplane(\"XY\").box(hook_thickness, hook_width, beam_top - plate_thickness + 0.5, centered=False)"
                ".translate((beam_x - hook_thickness, -hook_width / 2, plate_thickness - 0.5))\n"
            )
        else:
            s += (
                "beam = (\n    cq.Workplane(\"XY\")\n    .workplane(offset=plate_thickness - 0.5)\n"
                "    .center(beam_x - hook_thickness / 2, 0)\n    .rect(hook_thickness, hook_width)\n"
                "    .extrude(beam_top - plate_thickness + 0.5)\n)\n"
            )
        s += (
            "tip_x = plate_length / 2 + lip_reach\n"
            "lip = (\n    cq.Workplane(\"XZ\")\n    .polyline([\n        (beam_x - 0.5, lip_bottom_z),\n"
            "        (tip_x, lip_bottom_z),\n        (tip_x, lip_bottom_z + 0.35 * lip_height),\n"
            "        (beam_x - 0.5, beam_top),\n    ])\n    .close()\n    .extrude(hook_width / 2, both=True)\n)\n"
            "tongue = cq.Workplane(\"XY\").box(tongue_length + 1, tongue_width, tongue_thickness, centered=False)"
            ".translate((-plate_length / 2 - tongue_length, -tongue_width / 2, plate_thickness - tongue_thickness))\n"
            "result = plate.union(beam).union(lip).union(tongue)\n"
        )
        return s

    def mating_program(self, m, d):
        H = m["window_z"] + m["window_height"] + 3
        return (
            "import cadquery as cq\n"
            + "".join(f"{k} = {v}\n" for k, v in m.items())
            + f"H = {r2(H)}\n"
            "outer = cq.Workplane(\"XY\").rect(opening_length + 2 * wall, opening_width + 2 * wall).extrude(H)\n"
            "opening = cq.Workplane(\"XY\").workplane(offset=-1).rect(opening_length, opening_width).extrude(H + 2)\n"
            "window = cq.Workplane(\"XY\").box(wall + 0.4, window_width, window_height, centered=False)"
            ".translate((opening_length / 2 - 0.2, -window_width / 2, window_z))\n"
            "slot = cq.Workplane(\"XY\").box(wall + 0.4, slot_width, slot_top + 1, centered=False)"
            ".translate((-opening_length / 2 - wall - 0.2, -slot_width / 2, -1))\n"
            "result = outer.cut(opening).cut(window).cut(slot)\n"
        )

    def fit_specs(self, m, d):
        L, W, t = m["opening_length"], m["opening_width"], m["recess_depth"]
        z0, z1 = 0.25 * t, 0.75 * t
        sw = m["slot_width"] / 2 + 0.6
        pts, nrm = [], []
        for sign in (1, -1):
            p, n = plane_points((sign * L / 2, -0.4 * W, z0), (0, 0.8 * W, 0), (0, 0, z1 - z0), 9, 2, (-sign, 0, 0))
            if sign < 0:
                keep = np.abs(p[:, 1]) > sw
                p, n = p[keep], n[keep]
            pts.append(p)
            nrm.append(n)
            p, n = plane_points((-0.4 * L, sign * W / 2, z0), (0.8 * L, 0, 0), (0, 0, z1 - z0), 11, 2, (0, -sign, 0))
            pts.append(p)
            nrm.append(n)
        perim = FitSpec("plate_in_opening", "slide", "outer", np.vstack(pts), np.vstack(nrm), 30.0,
                        "plate_length", "opening edges", 1, 0.5)
        x_w = L / 2
        hw = min(d["hook_width"], m["window_width"]) / 2
        lp, ln = plane_points((x_w + 0.15, -0.5 * hw, m["window_z"]), (0.45, 0, 0), (0, hw, 0), 3, 4, (0, 0, 1))
        seat = FitSpec("lip_on_catch", "seat", "z", lp, ln, 10.0, "lip_bottom_z", "snap hook catch", 1, 1.0)
        tw = min(d["tongue_width"], m["slot_width"]) / 2
        tp, tn = plane_points((-L / 2 - 0.2, -0.5 * tw, m["slot_top"]), (-0.6, 0, 0), (0, tw, 0), 3, 4, (0, 0, -1))
        tongue = FitSpec("tongue_in_slot", "seat", "z", tp, tn, 10.0, "plate_thickness", "tongue slot", -1, 1.0)
        return [perim, seat, tongue]

    def probes(self, d):
        return [
            Probe("plate_width", "extent", {"axis": 1}, d["plate_width"], 0.3, True),
            Probe("plate_thickness", "thickness", {"origin": [0, 0, -1], "dir": [0, 0, 1]}, d["plate_thickness"], 0.12),
            Probe("top_z", "extent_max", {"axis": 2}, d["lip_bottom_z"] + d["lip_height"], 0.3, False),
        ]

    def strength_checks(self, m, d):
        L = d["lip_bottom_z"] - d["plate_thickness"]
        xb = d["plate_length"] / 2 - d["hook_setback"]
        zmid = d["plate_thickness"] + 0.5 * L
        x_w = m["opening_length"] / 2
        return [StrengthCheck("snap_hook_strain", "cantilever_strain", {
            "t": Probe("beam_t", "thickness", {"origin": [xb + 3, 0, zmid], "dir": [-1, 0, 0]}, d["hook_thickness"], 0.1),
            "tip": Probe("tip", "ray", {"origin": [x_w + 6, 0, d["lip_bottom_z"] + 0.1], "dir": [-1, 0, 0]}, 0, 0),
        }, {"L": L, "y_from": "x_w + 6 - tip - x_w", "x_w_off": 6.0})]

    def load_case(self, m, d):
        L = d["lip_bottom_z"] - d["plate_thickness"]
        xb = d["plate_length"] / 2 - d["hook_setback"]
        overlap = d["lip_reach"] - (m["opening_length"] - d["plate_length"]) / 2
        return LoadCase("snap_insertion", [-1e3, -1e3, -1, 1e3, 1e3, d["plate_thickness"] + 0.05],
                        [xb - d["hook_thickness"] - 0.5, -1e3, d["lip_bottom_z"], 1e3, 1e3, 1e3],
                        "disp", [-max(overlap, 0.0), 0, 0], "strain",
                        crop_box=[xb - d["hook_thickness"] - 4, -d["hook_width"] / 2 - 3, -1, 1e3, d["hook_width"] / 2 + 3, 1e3],
                        h=max(0.3, d["hook_thickness"] / 2.5)) if L > 0 else None

    def feature_graph(self, m, d, o=None):
        return {"part_type": self.part_type, "frame": FRAME_FLAT + "; outer face on the bed",
                "body": {"type": "rounded_plate", "length_x": d["plate_length"], "width_y": d["plate_width"],
                         "thickness": d["plate_thickness"], "corner_radius": d["corner_radius"], "critical": True,
                         "fit": "slide", "mates_with": f"opening {m['opening_length']} x {m['opening_width']}"},
                "features": [
                    {"id": "snap_hook", "type": "cantilever_snap_hook", "at": "+x end, on top face",
                     "beam_outer_face_x": r2(d["plate_length"] / 2 - d["hook_setback"]),
                     "beam_thickness": d["hook_thickness"], "width": d["hook_width"],
                     "lip_catch_face_z": d["lip_bottom_z"], "lip_height": d["lip_height"],
                     "lip_tip_x": r2(d["plate_length"] / 2 + d["lip_reach"]), "lip_ramp": "on top side",
                     "critical": True, "fit": "seat", "mates_with": f"window in wall, bottom face at z={m['window_z']}"},
                    {"id": "tongue", "type": "tab", "at": "-x end, flush with top face", "length": d["tongue_length"],
                     "width": d["tongue_width"], "thickness": d["tongue_thickness"],
                     "mates_with": f"slot in wall, top at z={m['slot_top']}"}]}

    def break_program(self, o, rng):
        src = self.program(o, 0)
        src += ("\n# --- simulated breakage: snap hook snapped off at the root\n"
                f"stub = cq.Workplane(\"XY\").box(20, {o['hook_width'] + 2}, 50, centered=False)"
                f".translate(({o['plate_length'] / 2 - o['hook_setback'] - o['hook_thickness'] - 1}, "
                f"{-(o['hook_width'] + 2) / 2}, {o['plate_thickness'] + 0.4}))\nresult = result.cut(stub)\n")
        return src, {"type": "snap_hook_missing", "missing": ["snap_hook"]}


# --------------------------------------------------------------------------------------
class LBracket(Family):
    name = "l_bracket"
    split = "train"
    part_type = "L-shaped mounting bracket with screw holes"
    n_styles = 2

    def sample_mating(self, rng):
        sd = float(rng.choice([3.0, 3.5, 4.0, 5.0]))
        return {"screw_diameter": sd, "load_N": r2(rng.uniform(8, 35))}

    def sample_original(self, rng, m):
        sd = m["screw_diameter"]
        W = r2(rng.uniform(max(10, 2.5 * sd), 25))
        t = r2(rng.uniform(2.0, 5.0))
        L1 = r2(rng.uniform(20, 60))
        L2 = r2(rng.uniform(20, 60))
        nb, nv = int(rng.integers(1, 3)), int(rng.integers(1, 3))
        o = {
            "base_length": L1, "upright_length": L2, "thickness": t, "width": W,
            "inner_fillet": r2(rng.uniform(0.5, min(5, t * 1.2))),
            "hole_diameter": r2(sd + rng.uniform(0.2, 0.6)),
            "base_holes": float(nb), "upright_holes": float(nv),
            "base_hole_offset": r2(rng.uniform(t + sd + 1.5, max(t + sd + 1.6, L1 * 0.45))),
            "upright_hole_offset": r2(rng.uniform(t + sd + 1.5, max(t + sd + 1.6, L2 * 0.45))),
        }
        o["base_hole_spacing"] = r2(rng.uniform(sd + 3, max(sd + 3.1, L1 - o["base_hole_offset"] - sd - 1.5))) if nb == 2 else 0.0
        o["upright_hole_spacing"] = r2(rng.uniform(sd + 3, max(sd + 3.1, L2 - o["upright_hole_offset"] - sd - 1.5))) if nv == 2 else 0.0
        return o

    def derive_design(self, m, o, pr, material="PLA", bias=None):
        d = dict(o)
        a = self.allow("loose", "hole_h", m["screw_diameter"], pr, bias, "hole_diameter")
        d["hole_diameter"] = r2(m["screw_diameter"] + 2 * a)
        mat = MATERIALS[material]
        allow_s = mat.yield_mpa / SAFETY_FACTOR
        t = d["thickness"]
        while cantilever_stress(m["load_N"], d["base_length"], d["width"], t) > allow_s and t < 10:
            t += 0.25
        d["thickness"] = r2(t)
        d["inner_fillet"] = r2(max(d["inner_fillet"], 0.5 * t))
        return d

    def holes(self, d):
        hs = []
        for i in range(int(d["base_holes"])):
            hs.append(("y", d["base_hole_offset"] + i * d["base_hole_spacing"]))
        for i in range(int(d["upright_holes"])):
            hs.append(("x", d["upright_hole_offset"] + i * d["upright_hole_spacing"]))
        return hs

    def program(self, p, style=0):
        s = header("L-bracket: base leg along +X, upright leg along +Y, extruded along Z (lies on its side)",
                   FRAME_PROFILE)
        s += fmt_params([
            ("base_length", p["base_length"], ""),
            ("upright_length", p["upright_length"], ""),
            ("thickness", p["thickness"], "strength-critical"),
            ("width", p["width"], "extrusion depth"),
            ("inner_fillet", p["inner_fillet"], ""),
            ("hole_diameter", p["hole_diameter"], "critical: screw clearance"),
            ("base_hole_offset", p["base_hole_offset"], "critical: first base hole x"),
            ("base_hole_spacing", p["base_hole_spacing"], "critical"),
            ("upright_hole_offset", p["upright_hole_offset"], "critical: first upright hole y"),
            ("upright_hole_spacing", p["upright_hole_spacing"], "critical"),
            ("base_holes", int(p["base_holes"]), ""),
            ("upright_holes", int(p["upright_holes"]), ""),
        ]) + "\n"
        s += (
            "pts = [(0, 0), (base_length, 0), (base_length, thickness), (thickness, thickness),\n"
            "       (thickness, upright_length), (0, upright_length)]\n"
            "bracket = cq.Workplane(\"XY\").polyline(pts).close().extrude(width)\n"
            "bracket = bracket.edges(\"|Z\").edges(cq.selectors.NearestToPointSelector((thickness, thickness, width / 2)))"
            ".fillet(inner_fillet)\n"
        )
        if style == 0:
            s += (
                "for i in range(base_holes):\n    x = base_hole_offset + i * base_hole_spacing\n"
                "    bracket = bracket.cut(cq.Workplane(\"XZ\", origin=(x, 0, width / 2)).circle(hole_diameter / 2)"
                ".extrude(-thickness - 1).translate((0, -0.5, 0)))\n"
                "for i in range(upright_holes):\n    y = upright_hole_offset + i * upright_hole_spacing\n"
                "    bracket = bracket.cut(cq.Workplane(\"YZ\", origin=(-0.5, y, width / 2)).circle(hole_diameter / 2)"
                ".extrude(thickness + 1))\n"
            )
        else:
            s += (
                "for i in range(base_holes):\n    x = base_hole_offset + i * base_hole_spacing\n"
                "    hole = cq.Solid.makeCylinder(hole_diameter / 2, thickness + 2, cq.Vector(x, -1, width / 2), "
                "cq.Vector(0, 1, 0))\n    bracket = bracket.cut(cq.Workplane(\"XY\").add(hole))\n"
                "for i in range(upright_holes):\n    y = upright_hole_offset + i * upright_hole_spacing\n"
                "    hole = cq.Solid.makeCylinder(hole_diameter / 2, thickness + 2, cq.Vector(-1, y, width / 2), "
                "cq.Vector(1, 0, 0))\n    bracket = bracket.cut(cq.Workplane(\"XY\").add(hole))\n"
            )
        return s + "result = bracket\n"

    def mating_program(self, m, d):
        s = "import cadquery as cq\nr = %s\nparts = []\n" % (m["screw_diameter"] / 2)
        z = d["width"] / 2
        for ax, pos in self.holes(d):
            if ax == "y":
                s += f"parts.append(cq.Solid.makeCylinder(r, {d['thickness'] + 6}, cq.Vector({pos}, -3, {z}), cq.Vector(0, 1, 0)))\n"
            else:
                s += f"parts.append(cq.Solid.makeCylinder(r, {d['thickness'] + 6}, cq.Vector(-3, {pos}, {z}), cq.Vector(1, 0, 0)))\n"
        return s + "result = cq.Workplane(\"XY\").add(cq.Compound.makeCompound(parts))\n"

    def fit_specs(self, m, d):
        r, t, z = m["screw_diameter"] / 2, d["thickness"], d["width"] / 2
        P, N = [], []
        for ax, pos in self.holes(d):
            c = (pos, 0, z) if ax == "y" else (0, pos, z)
            p, n = cyl_points(c, ax, r, 0.2 * t, 0.8 * t, 16, 3)
            P.append(p)
            N.append(n)
        return [FitSpec("screws_in_holes", "loose", "hole_h", np.vstack(P), np.vstack(N), m["screw_diameter"],
                        "hole_diameter", "screw holes")]

    def probes(self, d):
        return [
            Probe("base_length", "extent", {"axis": 0}, d["base_length"], 0.3, True),
            Probe("upright_length", "extent", {"axis": 1}, d["upright_length"], 0.3, True),
            Probe("width", "extent", {"axis": 2}, d["width"], 0.3, False),
            Probe("thickness", "thickness", {"origin": [d["base_length"] - 1.0, -2, 1.0], "dir": [0, 1, 0]},
                  d["thickness"], 0.15, True),
        ]

    def strength_checks(self, m, d):
        bx = d["base_length"] - 1.0
        return [StrengthCheck("base_leg_bending", "cantilever_stress", {
            "t": Probe("t", "thickness", {"origin": [0.5 * (d["thickness"] + d["base_length"]), -2, 1.0], "dir": [0, 1, 0]},
                       d["thickness"], 0.1),
            "b": Probe("b", "extent", {"axis": 2}, d["width"], 0.1),
        }, {"F": m["load_N"], "L": d["base_length"], "_bx": bx})]

    def load_case(self, m, d):
        t = d["thickness"]
        return LoadCase("shelf_load", [-0.1, d["upright_length"] * 0.5, -0.1, t + 0.1, 1e3, 1e3],
                        [d["base_length"] - 4, t - 0.3, -0.1, 1e3, t + 0.1, 1e3], "force",
                        [0, -m["load_N"], 0], "stress")

    def feature_graph(self, m, d, o=None):
        holes = []
        for i, (ax, pos) in enumerate(self.holes(d)):
            holes.append({"id": f"hole{i}", "type": "through_hole", "axis": ax,
                          "in_leg": "base" if ax == "y" else "upright",
                          "center": [pos, None, r2(d["width"] / 2)] if ax == "y" else [None, pos, r2(d["width"] / 2)],
                          "diameter": d["hole_diameter"], "critical": True, "fit": "loose",
                          "mates_with": f"screw d={m['screw_diameter']}"})
        return {"part_type": self.part_type, "frame": FRAME_PROFILE,
                "body": {"type": "L_profile_extrusion", "base_length_x": d["base_length"],
                         "upright_length_y": d["upright_length"], "thickness": d["thickness"],
                         "extrusion_z": d["width"], "inner_fillet": d["inner_fillet"]},
                "load": {"force_N": m["load_N"], "at": "end of base leg, -y", "fixed": "upright leg (screwed to wall)"},
                "features": holes}

    def break_program(self, o, rng):
        src = self.program(o, 0)
        x = o["base_length"] * rng.uniform(0.55, 0.8)
        src += (f"\n# --- simulated breakage: base leg snapped at x={x:.1f}\n"
                f"cut = cq.Workplane(\"XY\").box(200, 50, 200, centered=False).translate(({x:.2f}, -10, -10))\n"
                "result = result.cut(cut)\n")
        return src, {"type": "leg_truncated", "x": x, "missing": ["base_leg_end"]}


# --------------------------------------------------------------------------------------
class CableClip(Family):
    name = "cable_clip"
    split = "train"
    part_type = "snap-in cable / tube clip with screw tab"
    n_styles = 2

    def sample_mating(self, rng):
        return {"cable_diameter": r2(rng.uniform(3, 14)), "screw_diameter": float(rng.choice([2.5, 3.0, 3.5, 4.0]))}

    def sample_original(self, rng, m):
        dc = m["cable_diameter"]
        wall = r2(rng.uniform(1.2, 2.4))
        return {
            "inner_diameter": r2(dc + rng.uniform(-0.05, 0.1)),
            "wall": wall,
            "aperture": r2(dc * rng.uniform(0.70, 0.86)),
            "width": r2(rng.uniform(6, 14)),
            "base_thickness": r2(rng.uniform(1.6, 3.0)),
            "tab_length": r2(m["screw_diameter"] + rng.uniform(4, 8)),
            "hole_diameter": r2(m["screw_diameter"] + rng.uniform(0.2, 0.5)),
        }

    def derive_design(self, m, o, pr, material="PLA", bias=None):
        d = dict(o)
        a = self.allow("transition", "hole", m["cable_diameter"], pr, bias, "inner_diameter")
        d["inner_diameter"] = r2(m["cable_diameter"] + 2 * a)
        b = self.allow("loose", "hole_h", m["screw_diameter"], pr, bias, "hole_diameter")
        d["hole_diameter"] = r2(m["screw_diameter"] + 2 * b)
        # strength rule: arm strain during snap-in under allowable -> widen aperture / thin wall
        allow_e = MATERIALS[material].allow_strain_snap * 0.9
        ri = d["inner_diameter"] / 2
        for _ in range(60):
            rm = ri + d["wall"] / 2
            y = max(m["cable_diameter"] - d["aperture"], 0) / 2
            if snap_strain(d["wall"], y, rm * math.pi / 2) <= allow_e:
                break
            if d["aperture"] < 0.9 * m["cable_diameter"]:
                d["aperture"] = r2(d["aperture"] + 0.05)
            elif d["wall"] > 1.0:
                d["wall"] = r2(d["wall"] - 0.05)
            else:
                break
        return d

    def program(self, p, style=0):
        s = header("Snap-in cable clip: C-ring opening to +Y, screw tab at +X", FRAME_PROFILE).replace(
            "import cadquery as cq\n", "import math\n\nimport cadquery as cq\n")
        s += fmt_params([
            ("inner_diameter", p["inner_diameter"], "critical: snug on cable"),
            ("wall", p["wall"], "ring wall (snap stiffness)"),
            ("aperture", p["aperture"], "critical: snap opening width"),
            ("width", p["width"], "extrusion length along cable"),
            ("base_thickness", p["base_thickness"], ""),
            ("tab_length", p["tab_length"], ""),
            ("hole_diameter", p["hole_diameter"], "critical: screw clearance"),
        ]) + "\n"
        s += (
            "inner_r = inner_diameter / 2\nouter_r = inner_r + wall\nring_y = base_thickness + inner_r\n"
            "ring = cq.Workplane(\"XY\").center(0, ring_y).circle(outer_r).extrude(width)\n"
            "tab = cq.Workplane(\"XY\").center(tab_length / 2, base_thickness / 2)"
            ".rect(2 * outer_r + tab_length, base_thickness).extrude(width)\n"
            "bore = cq.Workplane(\"XY\").center(0, ring_y).circle(inner_r).extrude(width)\n"
            "half_angle = math.asin(aperture / (2 * inner_r))  # radial mouth: chord at the inner radius = aperture\n"
            "far = (outer_r + 1) / math.cos(half_angle)\n"
            "mouth = (\n    cq.Workplane(\"XY\")\n    .polyline([(0, ring_y), (far * math.sin(half_angle), ring_y + far * math.cos(half_angle)),\n"
            "               (-far * math.sin(half_angle), ring_y + far * math.cos(half_angle))])\n    .close()\n    .extrude(width)\n)\n"
        )
        if style == 0:
            s += ("screw = cq.Solid.makeCylinder(hole_diameter / 2, base_thickness + 2, "
                  "cq.Vector(outer_r + tab_length / 2, -1, width / 2), cq.Vector(0, 1, 0))\n"
                  "result = ring.union(tab).cut(bore).cut(mouth).cut(cq.Workplane(\"XY\").add(screw))\n")
        else:
            s += ("clip = ring.union(tab).cut(bore).cut(mouth)\n"
                  "screw = (\n    cq.Workplane(\"XZ\", origin=(outer_r + tab_length / 2, 0, width / 2))\n"
                  "    .circle(hole_diameter / 2)\n    .extrude(-base_thickness - 2)\n    .translate((0, -1, 0))\n)\n"
                  "result = clip.cut(screw)\n")
        return s

    def mating_program(self, m, d):
        ry = d["base_thickness"] + d["inner_diameter"] / 2
        ro = d["inner_diameter"] / 2 + d["wall"]
        return ("import cadquery as cq\n"
                f"cable = cq.Workplane(\"XY\").workplane(offset=-5).center(0, {r2(ry)}).circle({m['cable_diameter'] / 2})"
                f".extrude({d['width'] + 10})\n"
                f"screw = cq.Solid.makeCylinder({m['screw_diameter'] / 2}, {d['base_thickness'] + 6}, "
                f"cq.Vector({r2(ro + d['tab_length'] / 2)}, -3, {d['width'] / 2}), cq.Vector(0, 1, 0))\n"
                "result = cable.add(screw)\n"
                "result = cq.Workplane(\"XY\").add(cq.Compound.makeCompound([cable.val(), screw]))\n")

    def fit_specs(self, m, d):
        ry = d["base_thickness"] + d["inner_diameter"] / 2
        rc = m["cable_diameter"] / 2
        W = d["width"]
        half_a = d["aperture"] / 2 + 0.4
        pts, nrm = cyl_points((0, ry, 0), "z", rc, 0.15 * W, 0.85 * W, 32, 4,
                              keep=lambda p: not (abs(p[0]) < half_a and p[1] > ry))
        ro = d["inner_diameter"] / 2 + d["wall"]
        sp, sn = cyl_points((ro + d["tab_length"] / 2, 0, W / 2), "y", m["screw_diameter"] / 2,
                            0.2 * d["base_thickness"], 0.8 * d["base_thickness"], 16, 3)
        return [FitSpec("clip_on_cable", "transition", "hole", pts, nrm, m["cable_diameter"], "inner_diameter", "cable"),
                FitSpec("screw_in_tab", "loose", "hole_h", sp, sn, m["screw_diameter"], "hole_diameter", "screw hole")]

    def probes(self, d):
        ry = d["base_thickness"] + d["inner_diameter"] / 2
        ro = d["inner_diameter"] / 2 + d["wall"]
        return [
            Probe("width", "extent", {"axis": 2}, d["width"], 0.2, True),
            Probe("overall_height", "extent", {"axis": 1},
                  ry + ro * math.cos(math.asin(min(d["aperture"] / d["inner_diameter"], 0.999))), 0.3, False),
            Probe("tab_reach", "extent", {"axis": 0}, 2 * ro + d["tab_length"], 0.4, False),
        ]

    def strength_checks(self, m, d):
        ri = d["inner_diameter"] / 2
        ry = d["base_thickness"] + ri
        rm = ri + d["wall"] / 2
        yc = ry + math.sqrt(max(ri * ri - (d["aperture"] / 2) ** 2, 0.0)) + 0.02
        return [StrengthCheck("arm_snap_strain", "cantilever_strain", {
            "t": Probe("wall", "thickness", {"origin": [-(ri + d["wall"] + 3), ry, d["width"] / 2], "dir": [1, 0, 0]},
                       d["wall"], 0.1),
            "half_aperture": Probe("half_ap", "ray", {"origin": [0, yc, d["width"] / 2], "dir": [1, 0, 0]}, 0, 0),
        }, {"L": rm * math.pi / 2, "deflect_target": m["cable_diameter"]})]

    def load_case(self, m, d):
        ri = d["inner_diameter"] / 2
        ry = d["base_thickness"] + ri
        y = max(m["cable_diameter"] - d["aperture"], 0) / 2
        return LoadCase("snap_open", [-1e3, -1, -1, 1e3, d["base_thickness"] + 0.2, 1e3],
                        [0.0, ry + 0.6 * ri, -1, 1e3, 1e3, 1e3], "disp", [y, 0, 0], "strain")

    def feature_graph(self, m, d, o=None):
        return {"part_type": self.part_type, "frame": FRAME_PROFILE,
                "body": {"type": "C_ring_on_base_tab", "extrusion_z": d["width"], "base_thickness": d["base_thickness"]},
                "features": [
                    {"id": "ring", "type": "c_ring_snap", "axis": "z",
                     "center": [0, r2(d["base_thickness"] + d["inner_diameter"] / 2)], "inner_diameter": d["inner_diameter"],
                     "wall": d["wall"], "opening": "+y", "aperture_width": d["aperture"], "critical": True,
                     "fit": "transition", "mates_with": f"cable d={m['cable_diameter']}"},
                    {"id": "tab", "type": "flange_tab", "extends": "+x", "length": d["tab_length"],
                     "thickness": d["base_thickness"]},
                    {"id": "screw_hole", "type": "through_hole", "axis": "y", "in": "tab",
                     "center": [r2(d["inner_diameter"] / 2 + d["wall"] + d["tab_length"] / 2), None, r2(d["width"] / 2)],
                     "diameter": d["hole_diameter"], "critical": True, "fit": "loose",
                     "mates_with": f"screw d={m['screw_diameter']}"}]}

    def break_program(self, o, rng):
        src = self.program(o, 0)
        ri = o["inner_diameter"] / 2
        ry = o["base_thickness"] + ri
        side = -1 if rng.random() < 0.5 else 1
        src += (f"\n# --- simulated breakage: one snap arm broken off ({'left' if side < 0 else 'right'})\n"
                f"cut = cq.Workplane(\"XY\").center({side * (ri + 5)}, {ry + ri * 0.3 + 5}).rect(10, 10).extrude({o['width'] + 1})\n"
                "result = result.cut(cut)\n")
        return src, {"type": "arm_missing", "side": side, "missing": ["c_ring_arm"], "symmetry": "mirror_x"}


# --------------------------------------------------------------------------------------
class ShelfPeg(Family):
    name = "shelf_peg"
    split = "train"
    part_type = "shelf support peg (pin into cabinet hole, support lip)"
    n_styles = 2

    def sample_mating(self, rng):
        return {"hole_diameter": r2(float(rng.choice([5.0, 6.0, 6.35, 7.0, 8.0])) + rng.uniform(0.0, 0.12)),
                "load_N": r2(rng.uniform(5, 25))}

    def sample_original(self, rng, m):
        h = m["hole_diameter"]
        return {
            "pin_diameter": r2(h - rng.uniform(0.0, 0.1)),
            "pin_length": r2(rng.uniform(6, 12)),
            "pin_flat": r2(rng.uniform(0.3, 0.8)),
            "flange_thickness": r2(rng.uniform(1.5, 3.0)),
            "flange_height": r2(h + rng.uniform(2, 6)),
            "width": r2(rng.uniform(max(8, h + 2), 14)),
            "lip_length": r2(rng.uniform(6, 15)),
            "lip_thickness": r2(rng.uniform(2.0, 4.0)),
        }

    def derive_design(self, m, o, pr, material="PLA", bias=None):
        d = dict(o)
        a = self.allow("transition", "outer_h", m["hole_diameter"], pr, bias, "pin_diameter")
        d["pin_diameter"] = r2(m["hole_diameter"] - 2 * a)
        allow_s = MATERIALS[material].yield_mpa / SAFETY_FACTOR
        t = d["lip_thickness"]
        while cantilever_stress(m["load_N"], d["lip_length"], d["width"], t) > allow_s and t < 8:
            t += 0.25
        d["lip_thickness"] = r2(t)
        return d

    def program(self, p, style=0):
        s = header("Shelf support peg: pin along +X with flat on the bed, flange, support lip toward -X", FRAME_FLAT)
        s += fmt_params([
            ("pin_diameter", p["pin_diameter"], "critical: snug in cabinet hole"),
            ("pin_length", p["pin_length"], ""),
            ("pin_flat", p["pin_flat"], "flat on the underside for printing"),
            ("flange_thickness", p["flange_thickness"], ""),
            ("flange_height", p["flange_height"], ""),
            ("width", p["width"], ""),
            ("lip_length", p["lip_length"], ""),
            ("lip_thickness", p["lip_thickness"], "strength-critical"),
        ]) + "\n"
        s += "pin_z = pin_diameter / 2 - pin_flat  # pin axis height\n"
        s += ("lip = cq.Workplane(\"XY\").box(lip_length + 0.5, width, lip_thickness, centered=False)"
              ".translate((-lip_length, -width / 2, 0))\n"
              "flange = cq.Workplane(\"XY\").box(flange_thickness, width, flange_height, centered=False)"
              ".translate((0, -width / 2, 0))\n")
        if style == 0:
            s += ("pin = cq.Workplane(\"XY\").add(cq.Solid.makeCylinder(pin_diameter / 2, pin_length + 0.5, "
                  "cq.Vector(flange_thickness - 0.5, 0, pin_z), cq.Vector(1, 0, 0)))\n")
        else:
            s += ("pin = (\n    cq.Workplane(\"YZ\", origin=(flange_thickness - 0.5, 0, pin_z))\n"
                  "    .circle(pin_diameter / 2)\n    .extrude(pin_length + 0.5)\n)\n")
        s += ("bed_cut = cq.Workplane(\"XY\").box(500, 500, 50, centered=(True, True, False)).translate((0, 0, -50))\n"
              "result = lip.union(flange).union(pin).cut(bed_cut)\n")
        return s

    def mating_program(self, m, d):
        zc = d["pin_diameter"] / 2 - d["pin_flat"]
        x0 = d["flange_thickness"] + 0.02
        return ("import cadquery as cq\n"
                f"wall = cq.Workplane(\"XY\").box(15, 24, 24).translate(({x0 + 7.5}, 0, {zc}))\n"
                f"hole = cq.Solid.makeCylinder({m['hole_diameter'] / 2}, 20, cq.Vector({x0 - 1}, 0, {zc}), cq.Vector(1, 0, 0))\n"
                "result = wall.cut(cq.Workplane(\"XY\").add(hole))\n")

    def fit_specs(self, m, d):
        zc = d["pin_diameter"] / 2 - d["pin_flat"]
        x0 = d["flange_thickness"]
        rh = m["hole_diameter"] / 2
        pts, nrm = cyl_points((0, 0, zc), "x", rh, x0 + 0.15 * d["pin_length"], x0 + 0.85 * d["pin_length"], 32, 4,
                              outward=False, keep=lambda p: p[2] > zc - 0.45 * rh)
        return [FitSpec("pin_in_hole", "transition", "outer_h", pts, nrm, m["hole_diameter"], "pin_diameter",
                        "cabinet hole", -1, 0.5)]

    def probes(self, d):
        return [
            Probe("width", "extent", {"axis": 1}, d["width"], 0.25, False),
            Probe("overall_length", "extent", {"axis": 0},
                  d["lip_length"] + d["flange_thickness"] + d["pin_length"], 0.3, True),
            Probe("lip_thickness", "thickness", {"origin": [-d["lip_length"] / 2, 0, -1], "dir": [0, 0, 1]},
                  d["lip_thickness"], 0.15, True),
        ]

    def strength_checks(self, m, d):
        return [StrengthCheck("lip_bending", "cantilever_stress", {
            "t": Probe("t", "thickness", {"origin": [-d["lip_length"] / 2, 0, -1], "dir": [0, 0, 1]}, d["lip_thickness"], 0.1),
            "b": Probe("b", "extent", {"axis": 1}, d["width"], 0.1),
        }, {"F": m["load_N"], "L": d["lip_length"]})]

    def load_case(self, m, d):
        return LoadCase("shelf_on_lip", [d["flange_thickness"] + 0.5, -1e3, -1, 1e3, 1e3, 1e3],
                        [-1e3, -1e3, d["lip_thickness"] - 0.3, -d["lip_length"] + 3, 1e3, 1e3], "force",
                        [0, 0, -m["load_N"]], "stress")

    def feature_graph(self, m, d, o=None):
        return {"part_type": self.part_type, "frame": FRAME_FLAT,
                "body": {"type": "union_of_blocks"},
                "features": [
                    {"id": "lip", "type": "block", "x_range": [-d["lip_length"], 0], "width_y": d["width"],
                     "thickness_z": d["lip_thickness"], "load_N": m["load_N"]},
                    {"id": "flange", "type": "block", "x_range": [0, d["flange_thickness"]], "width_y": d["width"],
                     "height_z": d["flange_height"]},
                    {"id": "pin", "type": "pin", "axis": "x", "from_x": d["flange_thickness"], "length": d["pin_length"],
                     "diameter": d["pin_diameter"], "flat_on_bed": d["pin_flat"], "critical": True, "fit": "transition",
                     "mates_with": f"hole d={m['hole_diameter']}"}]}

    def break_program(self, o, rng):
        src = self.program(o, 0)
        cut_x = o["flange_thickness"] + o["pin_length"] * rng.uniform(0.2, 0.6)
        src += (f"\n# --- simulated breakage: pin sheared at x={cut_x:.1f}\n"
                f"cut = cq.Workplane(\"XY\").box(50, 50, 50, centered=(False, True, False)).translate(({cut_x:.2f}, 0, -1))\n"
                "result = result.cut(cut)\n")
        return src, {"type": "pin_sheared", "missing": ["pin_end"]}


# --------------------------------------------------------------------------------------
class WallHook(Family):
    name = "hook"
    split = "val"
    part_type = "J-shaped wall hook with screw holes"
    n_styles = 1

    def sample_mating(self, rng):
        return {"screw_diameter": float(rng.choice([3.0, 3.5, 4.0])), "load_N": r2(rng.uniform(5, 30))}

    def sample_original(self, rng, m):
        t = r2(rng.uniform(2.5, 5))
        hb = r2(rng.uniform(30, 60))
        return {"back_height": hb, "thickness": t, "width": r2(rng.uniform(10, 20)),
                "reach": r2(rng.uniform(15, 35)), "tip_height": r2(rng.uniform(8, 18)),
                "inner_fillet": r2(rng.uniform(1, 4)),
                "hole_diameter": r2(m["screw_diameter"] + rng.uniform(0.2, 0.5)),
                "hole1_y": r2(hb * rng.uniform(0.45, 0.6)), "hole2_y": r2(hb * rng.uniform(0.8, 0.9))}

    def derive_design(self, m, o, pr, material="PLA", bias=None):
        d = dict(o)
        a = self.allow("loose", "hole_h", m["screw_diameter"], pr, bias, "hole_diameter")
        d["hole_diameter"] = r2(m["screw_diameter"] + 2 * a)
        allow_s = MATERIALS[material].yield_mpa / SAFETY_FACTOR
        t = d["thickness"]
        while cantilever_stress(m["load_N"], d["reach"], d["width"], t) > allow_s and t < 10:
            t += 0.25
        d["thickness"] = r2(t)
        return d

    def program(self, p, style=0):
        s = header("J wall hook: back plate along +Y at x in [0, t], arm along +X at the bottom, tip up", FRAME_PROFILE)
        s += fmt_params([
            ("back_height", p["back_height"], ""), ("thickness", p["thickness"], "strength-critical"),
            ("width", p["width"], "extrusion"), ("reach", p["reach"], ""), ("tip_height", p["tip_height"], ""),
            ("inner_fillet", p["inner_fillet"], ""), ("hole_diameter", p["hole_diameter"], "critical: screw clearance"),
            ("hole1_y", p["hole1_y"], "critical"), ("hole2_y", p["hole2_y"], "critical"),
        ]) + "\n"
        s += ("pts = [(0, 0), (reach, 0), (reach, tip_height), (reach - thickness, tip_height),\n"
              "       (reach - thickness, thickness), (thickness, thickness), (thickness, back_height), (0, back_height)]\n"
              "hook = cq.Workplane(\"XY\").polyline(pts).close().extrude(width)\n"
              "hook = hook.edges(\"|Z\").edges(cq.selectors.NearestToPointSelector((thickness, thickness, width / 2)))"
              ".fillet(inner_fillet)\n"
              "for hy in (hole1_y, hole2_y):\n"
              "    hole = cq.Solid.makeCylinder(hole_diameter / 2, thickness + 2, cq.Vector(-1, hy, width / 2), cq.Vector(1, 0, 0))\n"
              "    hook = hook.cut(cq.Workplane(\"XY\").add(hole))\n"
              "result = hook\n")
        return s

    def mating_program(self, m, d):
        r = m["screw_diameter"] / 2
        return ("import cadquery as cq\n"
                f"a = cq.Solid.makeCylinder({r}, {d['thickness'] + 6}, cq.Vector(-3, {d['hole1_y']}, {d['width'] / 2}), cq.Vector(1, 0, 0))\n"
                f"b = cq.Solid.makeCylinder({r}, {d['thickness'] + 6}, cq.Vector(-3, {d['hole2_y']}, {d['width'] / 2}), cq.Vector(1, 0, 0))\n"
                "result = cq.Workplane(\"XY\").add(cq.Compound.makeCompound([a, b]))\n")

    def fit_specs(self, m, d):
        P, N = [], []
        for hy in (d["hole1_y"], d["hole2_y"]):
            p, n = cyl_points((0, hy, d["width"] / 2), "x", m["screw_diameter"] / 2, 0.2 * d["thickness"],
                              0.8 * d["thickness"], 16, 3)
            P.append(p)
            N.append(n)
        return [FitSpec("screws_in_holes", "loose", "hole_h", np.vstack(P), np.vstack(N), m["screw_diameter"],
                        "hole_diameter", "screw holes")]

    def probes(self, d):
        return [Probe("back_height", "extent", {"axis": 1}, d["back_height"], 0.3),
                Probe("reach", "extent", {"axis": 0}, d["reach"], 0.3),
                Probe("width", "extent", {"axis": 2}, d["width"], 0.3, False),
                Probe("thickness", "thickness", {"origin": [d["reach"] / 2 + d["thickness"] / 2, -2, 1], "dir": [0, 1, 0]},
                      d["thickness"], 0.15)]

    def strength_checks(self, m, d):
        return [StrengthCheck("arm_bending", "cantilever_stress", {
            "t": Probe("t", "thickness", {"origin": [d["reach"] / 2 + d["thickness"] / 2, -2, 1], "dir": [0, 1, 0]},
                       d["thickness"], 0.1),
            "b": Probe("b", "extent", {"axis": 2}, d["width"], 0.1)}, {"F": m["load_N"], "L": d["reach"]})]

    def load_case(self, m, d):
        return LoadCase("hang_load", [-0.1, d["back_height"] * 0.4, -0.1, d["thickness"] + 0.1, 1e3, 1e3],
                        [d["reach"] - d["thickness"] - 2, d["thickness"] - 0.3, -0.1, d["reach"] - d["thickness"] + 0.1,
                         d["thickness"] + 0.1, 1e3], "force", [0, -m["load_N"], 0], "stress")

    def feature_graph(self, m, d, o=None):
        return {"part_type": self.part_type, "frame": FRAME_PROFILE,
                "body": {"type": "J_profile_extrusion", "back_height_y": d["back_height"], "reach_x": d["reach"],
                         "tip_height_y": d["tip_height"], "thickness": d["thickness"], "extrusion_z": d["width"],
                         "inner_fillet": d["inner_fillet"]},
                "load": {"force_N": m["load_N"], "at": "inside of the hook arm, -y", "fixed": "back plate"},
                "features": [{"id": f"hole{i}", "type": "through_hole", "axis": "x", "in": "back plate",
                              "center": [None, y, r2(d["width"] / 2)], "diameter": d["hole_diameter"], "critical": True,
                              "fit": "loose", "mates_with": f"screw d={m['screw_diameter']}"}
                             for i, y in enumerate((d["hole1_y"], d["hole2_y"]))]}

    def break_program(self, o, rng):
        src = self.program(o)
        src += ("\n# --- simulated breakage: hook tip broken\n"
                f"cut = cq.Workplane(\"XY\").box(50, 50, 100, centered=False).translate(({o['reach'] - o['thickness'] - 0.5}, "
                f"{o['thickness'] + 0.5}, -10))\nresult = result.cut(cut)\n")
        return src, {"type": "tip_missing", "missing": ["tip"]}


# --------------------------------------------------------------------------------------
class HingeLeaf(Family):
    name = "hinge_leaf"
    split = "test"
    part_type = "hinge leaf with two knuckles and screw holes"
    n_styles = 1

    def sample_mating(self, rng):
        pin = r2(float(rng.choice([2.0, 2.5, 3.0, 4.0])) - rng.uniform(0, 0.04))
        return {"pin_diameter": pin, "screw_diameter": float(rng.choice([2.5, 3.0, 3.5])),
                "leaf_span": r2(rng.uniform(25, 50)), "middle_knuckle_length": r2(rng.uniform(8, 16))}

    def sample_original(self, rng, m):
        kr = r2(m["pin_diameter"] / 2 + rng.uniform(1.5, 3.0))
        span = m["leaf_span"]
        g = rng.uniform(0.1, 0.3)
        return {"leaf_span": span, "leaf_width": r2(rng.uniform(12, 25)), "thickness": r2(rng.uniform(1.8, min(3.5, 2 * kr - 0.5))),
                "knuckle_radius": kr, "pin_hole_diameter": r2(m["pin_diameter"] + rng.uniform(0.1, 0.3)),
                "knuckle_gap_length": r2(m["middle_knuckle_length"] + 2 * g), "relief": r2(rng.uniform(0.3, 0.6)),
                "hole_diameter": r2(m["screw_diameter"] + rng.uniform(0.2, 0.5)),
                "hole_x": r2(rng.uniform(0.55, 0.7)), "hole_spacing": r2(span * rng.uniform(0.45, 0.65))}

    def derive_design(self, m, o, pr, material="PLA", bias=None):
        d = dict(o)
        a = self.allow("clearance", "hole_h", m["pin_diameter"], pr, bias, "pin_hole_diameter")
        d["pin_hole_diameter"] = r2(m["pin_diameter"] + 2 * a)
        g = self.allow("clearance", "outer", 10.0, pr, bias, "knuckle_gap_length")
        d["knuckle_gap_length"] = r2(m["middle_knuckle_length"] + 2 * g)
        b = self.allow("loose", "hole", m["screw_diameter"], pr, bias, "hole_diameter")
        d["hole_diameter"] = r2(m["screw_diameter"] + 2 * b)
        return d

    def _hole_xy(self, d):
        x = d["knuckle_radius"] + d["relief"] + d["hole_x"] * d["leaf_width"]
        return [(r2(x), r2(-d["hole_spacing"] / 2)), (r2(x), r2(d["hole_spacing"] / 2))]

    def program(self, p, style=0):
        s = header("Hinge leaf: knuckle axis along Y at x=0, z=knuckle_radius; leaf toward +X lying on the bed", FRAME_FLAT)
        s += fmt_params([
            ("leaf_span", p["leaf_span"], "length along the hinge axis (Y)"),
            ("leaf_width", p["leaf_width"], "leaf extent in +X beyond the knuckle"),
            ("thickness", p["thickness"], ""),
            ("knuckle_radius", p["knuckle_radius"], ""),
            ("pin_hole_diameter", p["pin_hole_diameter"], "critical: clearance on hinge pin"),
            ("knuckle_gap_length", p["knuckle_gap_length"], "critical: gap for the mating middle knuckle"),
            ("relief", p["relief"], "radial clearance around mating knuckle"),
            ("hole_diameter", p["hole_diameter"], "critical: screw clearance"),
            ("hole_x_fraction", p["hole_x"], "screw hole position across the leaf"),
            ("hole_spacing", p["hole_spacing"], "critical"),
        ]) + "\n"
        s += ("leaf_x0 = knuckle_radius + relief\n"
              "leaf = cq.Workplane(\"XY\").box(leaf_width + leaf_x0, leaf_span, thickness, centered=False)"
              ".translate((0, -leaf_span / 2, 0))\n"
              "knuckle = cq.Workplane(\"XY\").add(cq.Solid.makeCylinder(knuckle_radius, leaf_span, "
              "cq.Vector(0, -leaf_span / 2, knuckle_radius), cq.Vector(0, 1, 0)))\n"
              "body = leaf.union(knuckle)\n"
              "gap = cq.Workplane(\"XY\").box(2 * knuckle_radius + 2, knuckle_gap_length, 2 * knuckle_radius + 2)"
              ".translate((leaf_x0 - knuckle_radius - 1, 0, knuckle_radius))\n"
              "body = body.cut(gap)\n"
              "pin_hole = cq.Workplane(\"XY\").add(cq.Solid.makeCylinder(pin_hole_diameter / 2, leaf_span + 2, "
              "cq.Vector(0, -leaf_span / 2 - 1, knuckle_radius), cq.Vector(0, 1, 0)))\n"
              "body = body.cut(pin_hole)\n"
              "hx = leaf_x0 + hole_x_fraction * leaf_width\n"
              "screws = cq.Workplane(\"XY\").workplane(offset=-1).pushPoints([(hx, -hole_spacing / 2), (hx, hole_spacing / 2)])"
              ".circle(hole_diameter / 2).extrude(thickness + 2)\n"
              "body = body.cut(screws)\n"
              "result = body\n")
        return s

    def mating_program(self, m, d):
        kr = d["knuckle_radius"]
        mk = m["middle_knuckle_length"]
        s = ("import cadquery as cq\nparts = []\n"
             f"parts.append(cq.Solid.makeCylinder({m['pin_diameter'] / 2}, {d['leaf_span'] + 4}, cq.Vector(0, {-d['leaf_span'] / 2 - 2}, {kr}), cq.Vector(0, 1, 0)))\n"
             f"k = cq.Solid.makeCylinder({kr}, {mk}, cq.Vector(0, {-mk / 2}, {kr}), cq.Vector(0, 1, 0))\n"
             f"k = k.cut(cq.Solid.makeCylinder({m['pin_diameter'] / 2 + 0.05}, {mk + 2}, cq.Vector(0, {-mk / 2 - 1}, {kr}), cq.Vector(0, 1, 0)))\n"
             "parts.append(k)\n")
        for (x, y) in self._hole_xy(d):
            s += f"parts.append(cq.Solid.makeCylinder({m['screw_diameter'] / 2}, {d['thickness'] + 6}, cq.Vector({x}, {y}, -3), cq.Vector(0, 0, 1)))\n"
        return s + "result = cq.Workplane(\"XY\").add(cq.Compound.makeCompound(parts))\n"

    def fit_specs(self, m, d):
        kr, span, mk = d["knuckle_radius"], d["leaf_span"], m["middle_knuckle_length"]
        gl = d["knuckle_gap_length"] / 2
        P, N = [], []
        for y0, y1 in ((-span / 2 + 0.2 * (span / 2 - gl), -gl - 0.3 * (span / 2 - gl)),
                       (gl + 0.3 * (span / 2 - gl), span / 2 - 0.2 * (span / 2 - gl))):
            p, n = cyl_points((0, 0, kr), "y", m["pin_diameter"] / 2, y0, y1, 16, 3)
            P.append(p)
            N.append(n)
        pin = FitSpec("knuckles_on_pin", "clearance", "hole_h", np.vstack(P), np.vstack(N), m["pin_diameter"],
                      "pin_hole_diameter", "hinge pin")
        ep, en = [], []
        for sgn in (1, -1):
            for rr in (0.5 * (m["pin_diameter"] / 2 + kr), 0.8 * kr):
                for a in np.linspace(0.3, 2 * np.pi - 0.3, 8):
                    ep.append([rr * np.cos(a), sgn * mk / 2, kr + rr * np.sin(a)])
                    en.append([0, sgn, 0])
        axial = FitSpec("knuckle_axial_gap", "clearance", "outer", np.array(ep), np.array(en), 10.0,
                        "knuckle_gap_length", "knuckle ends", 1, 0.5)
        sp, sn = [], []
        for (x, y) in self._hole_xy(d):
            p, n = cyl_points((x, y, 0), "z", m["screw_diameter"] / 2, 0.2 * d["thickness"], 0.8 * d["thickness"], 16, 3)
            sp.append(p)
            sn.append(n)
        screws = FitSpec("screws_in_leaf", "loose", "hole", np.vstack(sp), np.vstack(sn), m["screw_diameter"],
                         "hole_diameter", "screw holes")
        return [pin, axial, screws]

    def probes(self, d):
        return [Probe("leaf_span", "extent", {"axis": 1}, d["leaf_span"], 0.3),
                Probe("total_width", "extent", {"axis": 0}, 2 * d["knuckle_radius"] + d["relief"] + d["leaf_width"], 0.4, False),
                Probe("thickness", "thickness", {"origin": [d["knuckle_radius"] + d["relief"] + 1.0, 0, -1], "dir": [0, 0, 1]},
                      d["thickness"], 0.15)]

    def feature_graph(self, m, d, o=None):
        return {"part_type": self.part_type, "frame": FRAME_FLAT,
                "body": {"type": "plate_plus_cylinder", "plate": {"x_range": [0, r2(d["knuckle_radius"] + d["relief"] + d["leaf_width"])],
                                                                    "span_y": d["leaf_span"], "thickness": d["thickness"]},
                         "knuckle_cylinder": {"axis": "y", "center_xz": [0, d["knuckle_radius"]], "radius": d["knuckle_radius"]}},
                "features": [
                    {"id": "middle_gap", "type": "gap_cut", "centered_y": 0, "length_y": d["knuckle_gap_length"],
                     "removes": "knuckle and plate for x < knuckle_radius + relief", "relief": d["relief"],
                     "critical": True, "fit": "clearance", "mates_with": f"middle knuckle length {m['middle_knuckle_length']}"},
                    {"id": "pin_hole", "type": "through_hole", "axis": "y", "center_xz": [0, d["knuckle_radius"]],
                     "diameter": d["pin_hole_diameter"], "critical": True, "fit": "clearance",
                     "mates_with": f"pin d={m['pin_diameter']}"},
                    {"id": "screw_holes", "type": "through_holes", "axis": "z", "centers": self._hole_xy(d),
                     "diameter": d["hole_diameter"], "critical": True, "fit": "loose",
                     "mates_with": f"screws d={m['screw_diameter']}"}]}

    def break_program(self, o, rng):
        src = self.program(o)
        src += ("\n# --- simulated breakage: one outer knuckle broken off\n"
                f"cut = cq.Workplane(\"XY\").box({2 * o['knuckle_radius'] + 1}, {o['leaf_span'] / 2}, 20, centered=False)"
                f".translate(({-o['knuckle_radius'] - 0.5}, {o['knuckle_gap_length'] / 2 + 0.5}, {o['thickness'] + 0.01}))\n"
                "result = result.cut(cut)\n")
        return src, {"type": "knuckle_missing", "missing": ["knuckle_+y"], "symmetry": "mirror_y"}


# --------------------------------------------------------------------------------------
class RackClip(Family):
    name = "rack_clip"
    split = "test"
    part_type = "wire-rack clip with side C-channel and roller stud"
    n_styles = 1

    def sample_mating(self, rng):
        return {"wire_diameter": r2(rng.uniform(3.5, 6.5)), "roller_bore": r2(rng.uniform(3.0, 6.0))}

    def sample_original(self, rng, m):
        dw = m["wire_diameter"]
        sd = r2(m["roller_bore"] - rng.uniform(0.1, 0.3))
        cd = r2(dw + rng.uniform(-0.05, 0.1))
        lt = r2(rng.uniform(0.9, 1.5))
        return {"block_x": r2(sd + 2.0 + cd + rng.uniform(0.5, 3.0)), "block_y": r2(cd + lt + rng.uniform(1.5, 5)),
                "width": r2(rng.uniform(max(6, sd), 12)), "channel_diameter": cd, "aperture": r2(dw * rng.uniform(0.8, 0.9)),
                "lip_thickness": lt, "stud_diameter": sd, "stud_length": r2(rng.uniform(5, 10)), "stud_x": r2(sd / 2 + 0.6)}

    def derive_design(self, m, o, pr, material="PLA", bias=None):
        d = dict(o)
        a = self.allow("transition", "hole", m["wire_diameter"], pr, bias, "channel_diameter")
        d["channel_diameter"] = r2(m["wire_diameter"] + 2 * a)
        b = self.allow("clearance", "outer_h", m["roller_bore"], pr, bias, "stud_diameter")
        d["stud_diameter"] = r2(m["roller_bore"] - 2 * b)
        d["stud_diameter"] = min(d["stud_diameter"], d["width"])
        d["block_y"] = max(d["block_y"], r2(d["channel_diameter"] + d["lip_thickness"] + 1.5))
        d["stud_x"] = r2(d["stud_diameter"] / 2 + 0.6)
        d["block_x"] = max(d["block_x"], r2(d["stud_diameter"] + 1.2 + d["channel_diameter"] + 0.8))
        allow_e = MATERIALS[material].allow_strain_snap * 0.9
        for _ in range(60):
            ri = d["channel_diameter"] / 2
            L = (ri + d["lip_thickness"] / 2) * math.pi / 2
            y = max(m["wire_diameter"] - d["aperture"], 0) / 2
            if snap_strain(d["lip_thickness"], y, L) <= allow_e:
                break
            if d["aperture"] < 0.92 * m["wire_diameter"]:
                d["aperture"] = r2(d["aperture"] + 0.05)
            elif d["lip_thickness"] > 0.8:
                d["lip_thickness"] = r2(d["lip_thickness"] - 0.05)
            else:
                break
        return d

    def _geom(self, d):
        ri = d["channel_diameter"] / 2
        s = math.sqrt(max(ri * ri - (d["aperture"] / 2) ** 2, 0.01))
        cx = d["block_x"] - s
        cy = d["block_y"] - ri - d["lip_thickness"]
        return ri, cx, cy

    def program(self, p, style=0):
        s = header("Wire-rack clip: block with C-channel opening to +X (wire along Z), roller stud along +Y on top",
                   FRAME_PROFILE)
        s += fmt_params([
            ("block_x", p["block_x"], ""), ("block_y", p["block_y"], ""), ("width", p["width"], "extrusion along wire"),
            ("channel_diameter", p["channel_diameter"], "critical: snug on wire"),
            ("aperture", p["aperture"], "critical: snap mouth width at the +X face"),
            ("lip_thickness", p["lip_thickness"], "material above the channel (snap stiffness)"),
            ("stud_diameter", p["stud_diameter"], "critical: clearance in roller bore"),
            ("stud_length", p["stud_length"], ""), ("stud_x", p["stud_x"], "stud axis x position"),
        ]) + "\n"
        s += ("channel_r = channel_diameter / 2\n"
              "inset = (channel_r ** 2 - (aperture / 2) ** 2) ** 0.5\n"
              "cx, cy = block_x - inset, block_y - channel_r - lip_thickness\n"
              "block = cq.Workplane(\"XY\").rect(block_x, block_y, centered=False).extrude(width)\n"
              "channel = cq.Workplane(\"XY\").center(cx, cy).circle(channel_r).extrude(width)\n"
              "mouth = cq.Workplane(\"XY\").center((cx + block_x + 1) / 2, cy).rect(block_x + 1 - cx, aperture).extrude(width)\n"
              "stud = cq.Workplane(\"XY\").add(cq.Solid.makeCylinder(stud_diameter / 2, stud_length + 0.5, "
              "cq.Vector(stud_x, block_y - 0.5, stud_diameter / 2), cq.Vector(0, 1, 0)))\n"
              "result = block.cut(channel).cut(mouth).union(stud)\n")
        return s

    def mating_program(self, m, d):
        ri, cx, cy = self._geom(d)
        sx = d["stud_x"]
        zr = d["stud_diameter"] / 2
        L = d["stud_length"] - 1.0
        return ("import cadquery as cq\n"
                f"wire = cq.Solid.makeCylinder({m['wire_diameter'] / 2}, {d['width'] + 10}, cq.Vector({r2(cx)}, {r2(cy)}, -5), cq.Vector(0, 0, 1))\n"
                f"roller = cq.Solid.makeCylinder({m['roller_bore'] / 2 + 3}, {L}, cq.Vector({sx}, {d['block_y'] + 0.5}, {zr}), cq.Vector(0, 1, 0))\n"
                f"roller = roller.cut(cq.Solid.makeCylinder({m['roller_bore'] / 2}, {L + 2}, cq.Vector({sx}, {d['block_y'] - 0.5}, {zr}), cq.Vector(0, 1, 0)))\n"
                "result = cq.Workplane(\"XY\").add(cq.Compound.makeCompound([wire, roller]))\n")

    def fit_specs(self, m, d):
        ri, cx, cy = self._geom(d)
        W = d["width"]
        pts, nrm = cyl_points((cx, cy, 0), "z", m["wire_diameter"] / 2, 0.15 * W, 0.85 * W, 32, 4,
                              keep=lambda p: p[0] < cx + 0.2 * ri)
        sx, zr = d["stud_x"], d["stud_diameter"] / 2
        y0 = d["block_y"] + 0.5
        sp, sn = cyl_points((sx, 0, zr), "y", m["roller_bore"] / 2, y0 + 0.2 * (d["stud_length"] - 1),
                            y0 + 0.8 * (d["stud_length"] - 1), 24, 3, outward=False, keep=lambda p: p[2] > 0.3)
        return [FitSpec("clip_on_wire", "transition", "hole", pts, nrm, m["wire_diameter"], "channel_diameter", "rack wire"),
                FitSpec("roller_on_stud", "clearance", "outer_h", sp, sn, m["roller_bore"], "stud_diameter", "roller",
                        -1, 0.5)]

    def probes(self, d):
        return [Probe("block_x", "extent", {"axis": 0}, d["block_x"], 0.3),
                Probe("overall_y", "extent", {"axis": 1}, d["block_y"] + d["stud_length"], 0.3),
                Probe("width", "extent", {"axis": 2}, d["width"], 0.25)]

    def strength_checks(self, m, d):
        ri, cx, cy = self._geom(d)
        return [StrengthCheck("lip_snap_strain", "cantilever_strain", {
            "t": Probe("lip_t", "thickness", {"origin": [cx, d["block_y"] + 0.01, d["width"] - 0.3], "dir": [0, -1, 0]},
                       d["lip_thickness"], 0.1),
            "half_aperture": Probe("half_ap", "ray", {"origin": [d["block_x"] - 0.1, cy, d["width"] / 2], "dir": [0, 1, 0]}, 0, 0),
        }, {"L": (ri + d["lip_thickness"] / 2) * math.pi / 2, "deflect_target": m["wire_diameter"]})]

    def feature_graph(self, m, d, o=None):
        ri, cx, cy = self._geom(d)
        return {"part_type": self.part_type, "frame": FRAME_PROFILE,
                "body": {"type": "rectangular_block", "x_range": [0, d["block_x"]], "y_range": [0, d["block_y"]],
                         "extrusion_z": d["width"]},
                "features": [
                    {"id": "channel", "type": "c_channel_snap", "axis": "z", "center": [r2(cx), r2(cy)],
                     "diameter": d["channel_diameter"], "opening": "+x face", "aperture_width": d["aperture"],
                     "critical": True, "fit": "transition", "mates_with": f"wire d={m['wire_diameter']}"},
                    {"id": "stud", "type": "pin", "axis": "y", "from": "top face y=block_y",
                     "center_xz": [r2(d["stud_x"]), r2(d["stud_diameter"] / 2)],
                     "diameter": d["stud_diameter"], "length": d["stud_length"], "critical": True, "fit": "clearance",
                     "mates_with": f"roller bore d={m['roller_bore']}"}]}

    def break_program(self, o, rng):
        src = self.program(o)
        src += ("\n# --- simulated breakage: stud snapped off\n"
                f"cut = cq.Workplane(\"XY\").box(100, 50, 100, centered=(True, False, True)).translate((0, {o['block_y'] + 0.3}, 0))\n"
                "result = result.cut(cut)\n")
        return src, {"type": "stud_missing", "missing": ["stud"]}


# --------------------------------------------------------------------------------------
class EndCapRect(Family):
    name = "end_cap_rect"
    split = "test"
    part_type = "rectangular tube end plug (press fit inside tube)"
    n_styles = 1

    def sample_mating(self, rng):
        a = r2(rng.uniform(12, 40))
        b = r2(a * rng.uniform(0.4, 1.0))
        tw = r2(rng.uniform(1.0, 2.0))
        return {"tube_inner_x": a, "tube_inner_y": b, "tube_wall": tw, "tube_corner_r": r2(rng.uniform(0.3, 1.5))}

    def sample_original(self, rng, m):
        return self._plate(m, {"plug_x": r2(m["tube_inner_x"] + rng.uniform(0.05, 0.2)), "plug_y": r2(m["tube_inner_y"] + rng.uniform(0.05, 0.2)),
                "plug_depth": r2(rng.uniform(6, 15)), "plug_wall": r2(rng.uniform(1.2, 2.2)),
                "plate_overhang": r2(rng.uniform(0.0, 1.0)), "plate_thickness": r2(rng.uniform(2.0, 4.0)),
                "plate_corner_r": r2(rng.uniform(0.5, 2.5)), "plug_corner_r": r2(m["tube_corner_r"] + rng.uniform(0, 0.3))})

    def _plate(self, m, o):
        o["plate_x"] = r2(m["tube_inner_x"] + 2 * m["tube_wall"] + 2 * o["plate_overhang"])
        o["plate_y"] = r2(m["tube_inner_y"] + 2 * m["tube_wall"] + 2 * o["plate_overhang"])
        return o

    def derive_design(self, m, o, pr, material="PLA", bias=None):
        d = dict(o)
        from spare.features.tolerance import press_allowance

        span = min(m["tube_inner_x"], m["tube_inner_y"])
        # wall flex limit: eps = 12 t delta / L^2  ->  delta_max = eps L^2 / (12 t); expressed via an equivalent radius
        r_eq = span ** 2 / (12 * o["plug_wall"])
        a = press_allowance("outer", span, r_eq, MATERIALS[material].allow_strain_snap, pr, (bias or {}).get("plug_x", 0.0))
        d["plug_x"] = r2(m["tube_inner_x"] - 2 * a)
        d["plug_y"] = r2(m["tube_inner_y"] - 2 * a)
        return self._plate(m, d)

    def program(self, p, style=0):
        s = header("Rectangular tube end plug: cover plate on the bed, hollow plug rising along +Z",
                   "Z up = build direction, centred on the origin")
        s += fmt_params([
            ("plate_x", p["plate_x"], ""), ("plate_y", p["plate_y"], ""), ("plate_thickness", p["plate_thickness"], ""),
            ("plate_corner_r", p["plate_corner_r"], ""),
            ("plug_x", p["plug_x"], "critical: press fit in tube"), ("plug_y", p["plug_y"], "critical: press fit in tube"),
            ("plug_depth", p["plug_depth"], ""), ("plug_wall", p["plug_wall"], ""), ("plug_corner_r", p["plug_corner_r"], ""),
        ]) + "\n"
        s += ("plate = cq.Workplane(\"XY\").rect(plate_x, plate_y).extrude(plate_thickness).edges(\"|Z\").fillet(plate_corner_r)\n"
              "plug = (\n    cq.Workplane(\"XY\")\n    .workplane(offset=plate_thickness)\n    .rect(plug_x, plug_y)\n"
              "    .extrude(plug_depth)\n    .edges(\"|Z\").fillet(plug_corner_r)\n)\n"
              "hollow = (\n    cq.Workplane(\"XY\")\n    .workplane(offset=plate_thickness)\n"
              "    .rect(plug_x - 2 * plug_wall, plug_y - 2 * plug_wall)\n    .extrude(plug_depth + 1)\n)\n"
              "result = plate.union(plug).cut(hollow)\n")
        return s

    def mating_program(self, m, d):
        z0 = d["plate_thickness"] + 0.3
        return ("import cadquery as cq\n"
                f"outer = cq.Workplane(\"XY\").workplane(offset={z0}).rect({m['tube_inner_x'] + 2 * m['tube_wall']}, "
                f"{m['tube_inner_y'] + 2 * m['tube_wall']}).extrude({d['plug_depth'] + 15})\n"
                f"inner = cq.Workplane(\"XY\").workplane(offset={z0 - 1}).rect({m['tube_inner_x']}, {m['tube_inner_y']})"
                f".extrude({d['plug_depth'] + 17}).edges(\"|Z\").fillet({m['tube_corner_r']})\n"
                "result = outer.cut(inner)\n")

    def fit_specs(self, m, d):
        A, B = m["tube_inner_x"], m["tube_inner_y"]
        z0 = d["plate_thickness"] + 0.3 + 0.15 * d["plug_depth"]
        z1 = d["plate_thickness"] + 0.85 * d["plug_depth"]
        mx = max(m["tube_corner_r"], d["plug_corner_r"]) + 1.0
        P, N = [], []
        for sgn in (1, -1):
            p, n = plane_points((sgn * A / 2, -B / 2 + mx, z0), (0, B - 2 * mx, 0), (0, 0, z1 - z0), 6, 3, (-sgn, 0, 0))
            P.append(p)
            N.append(n)
            p, n = plane_points((-A / 2 + mx, sgn * B / 2, z0), (A - 2 * mx, 0, 0), (0, 0, z1 - z0), 8, 3, (0, -sgn, 0))
            P.append(p)
            N.append(n)
        fs = FitSpec("plug_in_tube", "press", "outer", np.vstack(P), np.vstack(N), min(A, B), "plug_x", "tube", -1, 0.5)
        fs.area = 2 * (A + B) * d["plug_depth"]  # type: ignore[attr-defined]
        return [fs]

    def probes(self, d):
        return [Probe("plate_x", "extent", {"axis": 0}, d["plate_x"], 0.3, False),
                Probe("plate_y", "extent", {"axis": 1}, d["plate_y"], 0.3, False),
                Probe("height", "extent", {"axis": 2}, d["plate_thickness"] + d["plug_depth"], 0.25),
                Probe("plate_thickness", "thickness", {"origin": [0, 0, -1], "dir": [0, 0, 1]}, d["plate_thickness"], 0.15)]

    def strength_checks(self, m, d):
        return [StrengthCheck("plug_wall_strain", "wall_flex", {
            "t": Probe("plug_wall", "thickness", {"origin": [-(d["plate_x"] / 2 + 3), 0, d["plate_thickness"] + d["plug_depth"] / 2],
                                                  "dir": [1, 0, 0]}, d["plug_wall"], 0.1)},
            {"fit_spec": 0, "span": min(m["tube_inner_x"], m["tube_inner_y"])})]

    def feature_graph(self, m, d, o=None):
        return {"part_type": self.part_type, "frame": "Z up = build direction, centred on origin, cover plate on the bed",
                "body": {"type": "rounded_plate", "x": d["plate_x"], "y": d["plate_y"], "thickness": d["plate_thickness"],
                         "corner_radius": d["plate_corner_r"]},
                "features": [{"id": "plug", "type": "hollow_rect_boss", "on": "top face", "x": d["plug_x"], "y": d["plug_y"],
                              "height": d["plug_depth"], "wall": d["plug_wall"], "corner_radius": d["plug_corner_r"],
                              "critical": True, "fit": "press",
                              "mates_with": f"tube inner {m['tube_inner_x']} x {m['tube_inner_y']}"}]}

    def break_program(self, o, rng):
        src = self.program(o)
        src += ("\n# --- simulated breakage: one plug wall cracked off\n"
                f"cut = cq.Workplane(\"XY\").box(10, {o['plug_y'] * 0.6}, 100, centered=(False, True, False))"
                f".translate(({o['plug_x'] / 2 - o['plug_wall'] - 0.2}, 0, {o['plate_thickness'] + 1}))\nresult = result.cut(cut)\n")
        return src, {"type": "plug_wall_missing", "missing": ["plug_wall_+x"], "symmetry": "mirror_x"}


_ = (FIT_BANDS, printed_deviation)
