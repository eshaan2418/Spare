"""Rotational families: spacer, knob with D-shaft, round end cap, grooved wheel."""

from __future__ import annotations

import math

import numpy as np

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

FRAME_Z = "axis = Z (build direction), base face on z=0, centred on the origin"


class Spacer(Family):
    name = "spacer"
    split = "train"
    part_type = "spacer / standoff ring"
    n_styles = 3

    def sample_mating(self, rng):
        bolt = float(rng.choice([2.5, 3, 4, 5, 6, 8]))
        return {"bolt_diameter": r2(bolt - rng.uniform(0.0, 0.08))}

    def sample_original(self, rng, m):
        b = m["bolt_diameter"]
        return {
            "outer_diameter": r2(b + rng.uniform(3, 9)),
            "bore_diameter": r2(b + rng.uniform(0.15, 0.45)),
            "height": r2(rng.uniform(3, 20)),
            "hex": float(rng.random() < 0.3),
            "chamfer": r2(rng.uniform(0.0, 0.6)),
        }

    def derive_design(self, m, o, pr, material="PLA", bias=None):
        d = dict(o)
        a = self.allow("clearance", "hole", m["bolt_diameter"], pr, bias, "bore_diameter")
        d["bore_diameter"] = r2(m["bolt_diameter"] + 2 * a)
        d["outer_diameter"] = max(d["outer_diameter"], r2(d["bore_diameter"] + 2.4))
        return d

    def program(self, p, style=0):
        hexa = p["hex"] > 0.5
        h = header("Spacer / standoff" + (" (hex)" if hexa else ""), FRAME_Z)
        prm = [
            ("outer_diameter", p["outer_diameter"], "across corners" if hexa else ""),
            ("bore_diameter", p["bore_diameter"], "critical: clearance fit on bolt"),
            ("height", p["height"], "critical: spacing"),
        ]
        if not hexa and p["chamfer"] > 0.05 and style == 0:
            prm.append(("chamfer", p["chamfer"], "outer edges"))
        h += fmt_params(prm) + "\n"
        if hexa:
            body = 'result = (\n    cq.Workplane("XY")\n    .polygon(6, outer_diameter)\n    .extrude(height)\n'
            body += '    .faces(">Z").workplane()\n    .hole(bore_diameter)\n)\n'
            return h + body
        if style == 0 and p["chamfer"] > 0.05:
            return h + (
                'result = (\n    cq.Workplane("XY")\n    .circle(outer_diameter / 2)\n    .extrude(height)\n'
                '    .edges().chamfer(chamfer)\n    .faces(">Z").workplane()\n    .hole(bore_diameter)\n)\n'
            )
        if style == 2:
            return h + (
                'ring = cq.Workplane("XY").circle(outer_diameter / 2).extrude(height)\n'
                'bore = cq.Workplane("XY").circle(bore_diameter / 2).extrude(height)\n'
                "result = ring.cut(bore)\n"
            )
        return h + (
            'result = (\n    cq.Workplane("XY")\n    .circle(outer_diameter / 2)\n'
            "    .circle(bore_diameter / 2)\n    .extrude(height)\n)\n"
        )

    def mating_program(self, m, d):
        return (
            "import cadquery as cq\n"
            f"bolt_diameter = {m['bolt_diameter']}\nheight = {d['height']}\n"
            'result = cq.Workplane("XY").workplane(offset=-3).circle(bolt_diameter / 2).extrude(height + 6)\n'
        )

    def fit_specs(self, m, d):
        h = d["height"]
        pts, nrm = cyl_points((0, 0, 0), "z", m["bolt_diameter"] / 2, 0.15 * h, 0.85 * h, 24, 5)
        return [FitSpec("bore_on_bolt", "clearance", "hole", pts, nrm, m["bolt_diameter"],
                        "bore_diameter", "bolt")]

    def probes(self, d):
        h = d["height"]
        pr = [
            Probe("height", "extent", {"axis": 2}, d["height"], 0.15),
            Probe("bore_diameter", "circle_d", {"axis": 2, "offset": h / 2, "near": [0, 0], "which": "inner"},
                  d["bore_diameter"], 0.12),
        ]
        if d["hex"] > 0.5:
            pr.append(Probe("across_corners", "extent", {"axis": 0}, d["outer_diameter"], 0.4, False))
        else:
            pr.append(Probe("outer_diameter", "extent", {"axis": 0}, d["outer_diameter"], 0.4, False))
        return pr

    def feature_graph(self, m, d, o=None):
        return {
            "part_type": self.part_type,
            "frame": FRAME_Z,
            "body": {"type": "hex_prism" if d["hex"] > 0.5 else "cylinder",
                     ("across_corners" if d["hex"] > 0.5 else "diameter"): d["outer_diameter"],
                     "height": d["height"], "critical": ["height"]},
            "features": [
                {"id": "bore", "type": "through_hole", "axis": "z", "at": [0, 0], "diameter": d["bore_diameter"],
                 "critical": True, "fit": "clearance", "mates_with": f"bolt d={m['bolt_diameter']}"},
            ] + ([{"id": "edge_chamfer", "type": "chamfer", "edges": "outer circular edges", "size": d["chamfer"]}]
                 if d["hex"] < 0.5 and d["chamfer"] > 0.05 else []),
        }

    def break_program(self, o, rng):
        # spacers crack radially: a wedge missing
        ang = float(rng.uniform(0, 360))
        src = self.program(o, 1)
        src += (
            f"\n# --- simulated breakage: radial wedge missing at {ang:.0f} deg\n"
            f'wedge = cq.Workplane("XY").polyline([(0, 0), ({o["outer_diameter"]}, -1.0), ({o["outer_diameter"]}, 1.0)])'
            f".close().extrude({o['height'] + 1}).rotate((0, 0, 0), (0, 0, 1), {ang:.1f})\n"
            "result = result.cut(wedge)\n"
        )
        return src, {"type": "radial_wedge", "angle_deg": ang}


class KnobDShaft(Family):
    name = "knob_dshaft"
    split = "train"
    part_type = "control knob with D-shaft bore"
    n_styles = 2

    def sample_mating(self, rng):
        sd = float(rng.choice([4.0, 5.0, 6.0, 6.0, 6.35]))
        sd = r2(sd - rng.uniform(0.0, 0.05))
        return {"shaft_diameter": sd, "shaft_flat": r2(sd * rng.uniform(0.70, 0.82))}

    def sample_original(self, rng, m):
        depth = r2(rng.uniform(7, 14))
        kd = r2(rng.uniform(max(14, m["shaft_diameter"] * 2.5), 40))
        kh = r2(depth + rng.uniform(2.5, 9))
        nr = int(rng.choice([0, 0, 12, 18, 24, 30]))
        return {
            "knob_diameter": kd,
            "knob_height": kh,
            "edge_fillet": r2(min(rng.uniform(0.5, 3.0), kh / 4, kd / 8)),
            "grip_ribs": float(nr),
            "rib_radius": r2(rng.uniform(0.5, 1.1)),
            "bore_diameter": r2(m["shaft_diameter"] + rng.uniform(0.0, 0.08)),
            "bore_flat": r2(m["shaft_flat"] + rng.uniform(0.0, 0.08)),
            "bore_depth": depth,
        }

    def derive_design(self, m, o, pr, material="PLA", bias=None):
        d = dict(o)
        a = self.allow("transition", "hole", m["shaft_diameter"], pr, bias, "bore_diameter")
        af = self.allow("transition", "hole", m["shaft_diameter"], pr, bias, "bore_flat")
        d["bore_diameter"] = r2(m["shaft_diameter"] + 2 * a)
        d["bore_flat"] = r2(m["shaft_flat"] + a + af)
        # DfAM rule (cosmetic geometry is flexible): grip teeth between grooves must be >= 1 mm wide
        if d["grip_ribs"] > 0:
            pitch = math.pi * d["knob_diameter"] / d["grip_ribs"]
            if pitch - 2 * d["rib_radius"] < 1.0:
                d["grip_ribs"] = float(max(6, int(math.pi * d["knob_diameter"] / (2 * d["rib_radius"] + 1.0))))
        return d

    def program(self, p, style=0):
        h = header("Control knob with D-shaft bore (knob face on the bed, bore opens upward)", FRAME_Z)
        nr = int(p["grip_ribs"])
        prm = [
            ("knob_diameter", p["knob_diameter"], ""),
            ("knob_height", p["knob_height"], ""),
            ("edge_fillet", p["edge_fillet"], "back edge"),
        ]
        if nr:
            prm += [("grip_ribs", nr, "number of grip grooves"), ("rib_radius", p["rib_radius"], "groove radius")]
        prm += [
            ("bore_diameter", p["bore_diameter"], "critical: transition fit on D-shaft"),
            ("bore_flat", p["bore_flat"], "critical: across-flat size of D-bore"),
            ("bore_depth", p["bore_depth"], ""),
        ]
        s = h + fmt_params(prm) + "\n"
        s += 'knob = cq.Workplane("XY").circle(knob_diameter / 2).extrude(knob_height)\n'
        s += 'knob = knob.faces(">Z").edges().fillet(edge_fillet)\n'
        if nr:
            s += (
                'grooves = (\n    cq.Workplane("XY")\n    .polarArray(knob_diameter / 2, 0, 360, grip_ribs)\n'
                "    .circle(rib_radius)\n    .extrude(knob_height)\n)\nknob = knob.cut(grooves)\n"
            )
        s += "flat_y = bore_flat - bore_diameter / 2  # flat face of the D at y = flat_y\n"
        if style == 0:
            s += (
                "d_bore = (\n    cq.Workplane(\"XY\")\n    .workplane(offset=knob_height - bore_depth)\n"
                "    .circle(bore_diameter / 2)\n    .extrude(bore_depth + 1)\n"
                "    .intersect(\n        cq.Workplane(\"XY\")\n        .center(0, (flat_y - bore_diameter) / 2)\n"
                "        .rect(2 * bore_diameter, flat_y + bore_diameter)\n        .extrude(knob_height + 1)\n    )\n)\n"
                "result = knob.cut(d_bore)\n"
            )
        else:
            s += (
                "half_chord = (bore_diameter ** 2 / 4 - flat_y ** 2) ** 0.5\n"
                "d_profile = (\n    cq.Workplane(\"XY\")\n    .workplane(offset=knob_height - bore_depth)\n"
                "    .moveTo(-half_chord, flat_y)\n    .lineTo(half_chord, flat_y)\n"
                "    .threePointArc((0, -bore_diameter / 2), (-half_chord, flat_y))\n    .close()\n"
                "    .extrude(bore_depth + 1)\n)\nresult = knob.cut(d_profile)\n"
            )
        return s

    def mating_program(self, m, d):
        zb = d["knob_height"] - d["bore_depth"]
        return (
            "import cadquery as cq\n"
            f"shaft_diameter = {m['shaft_diameter']}\nshaft_flat = {m['shaft_flat']}\n"
            f"z0 = {r2(zb + 0.5)}\nlength = {r2(d['bore_depth'] + 10)}\n"
            "fy = shaft_flat - shaft_diameter / 2\n"
            'shaft = cq.Workplane("XY").workplane(offset=z0).circle(shaft_diameter / 2).extrude(length)\n'
            'keep = cq.Workplane("XY").workplane(offset=z0).center(0, (fy - shaft_diameter) / 2)'
            ".rect(2 * shaft_diameter, fy + shaft_diameter).extrude(length)\n"
            "result = shaft.intersect(keep)\n"
        )

    def fit_specs(self, m, d):
        r = m["shaft_diameter"] / 2
        fy = m["shaft_flat"] - r
        z0 = d["knob_height"] - d["bore_depth"] + 1.2
        z1 = d["knob_height"] - 0.6
        pts, nrm = cyl_points((0, 0, 0), "z", r, z0, z1, 32, 5, keep=lambda p: p[1] < fy - 0.35)
        hc = math.sqrt(max(r * r - fy * fy, 0.01))
        fp, fn = plane_points((-0.6 * hc, fy, z0), (1.2 * hc, 0, 0), (0, 0, z1 - z0), 5, 5, (0, 1, 0))
        return [
            FitSpec("bore_on_shaft", "transition", "hole", pts, nrm, m["shaft_diameter"], "bore_diameter", "shaft"),
            FitSpec("flat_on_shaft_flat", "transition", "hole", fp, fn, m["shaft_diameter"], "bore_flat",
                    "shaft flat", 1, 1.0),
        ]

    def probes(self, d):
        return [
            Probe("knob_diameter", "extent", {"axis": 0}, d["knob_diameter"], 0.4, False),
            Probe("knob_height", "extent", {"axis": 2}, d["knob_height"], 0.2, True),
            Probe("bore_depth", "depth_from_top", {"at": [0, -d["bore_diameter"] / 4]}, d["bore_depth"], 0.25, True),
        ]

    def feature_graph(self, m, d, o=None):
        fe = [
            {"id": "d_bore", "type": "blind_d_bore", "axis": "z", "opens": "+z (top)", "at": [0, 0],
             "diameter": d["bore_diameter"], "across_flat": d["bore_flat"], "flat_side": "+y",
             "depth": d["bore_depth"], "critical": True, "fit": "transition",
             "mates_with": f"D-shaft d={m['shaft_diameter']} flat={m['shaft_flat']}"},
            {"id": "back_fillet", "type": "fillet", "edges": "top outer edge", "radius": d["edge_fillet"]},
        ]
        if d["grip_ribs"] > 0:
            fe.append({"id": "grip", "type": "radial_grooves", "count": int(d["grip_ribs"]),
                       "groove_radius": d["rib_radius"], "on": "outer cylinder"})
        return {"part_type": self.part_type, "frame": FRAME_Z,
                "body": {"type": "cylinder", "diameter": d["knob_diameter"], "height": d["knob_height"]},
                "features": fe}

    def break_program(self, o, rng):
        # a knob bore splits along the flat: crack + chunk of skirt missing
        src = self.program(o, 0)
        ang = float(rng.uniform(0, 360))
        R = o["knob_diameter"] / 2
        src += (
            f"\n# --- simulated breakage: chunk of the rim missing at {ang:.0f} deg\n"
            f'chunk = cq.Workplane("XY").center({R:.2f}, 0).rect({R * 0.8:.2f}, {R * 0.9:.2f})'
            f".extrude({o['knob_height'] + 1}).rotate((0, 0, 0), (0, 0, 1), {ang:.1f})\n"
            "result = result.cut(chunk)\n"
        )
        return src, {"type": "rim_chunk", "angle_deg": ang}


class CapRound(Family):
    name = "cap_round"
    split = "train"
    part_type = "round tube end cap (fits over tube)"
    n_styles = 2

    def sample_mating(self, rng):
        od = r2(float(rng.choice([8, 10, 12, 15, 16, 19, 20, 22, 25, 28, 32])) + rng.uniform(-0.15, 0.1))
        return {"tube_outer_diameter": od, "tube_wall": r2(rng.uniform(0.8, 2.0))}

    def sample_original(self, rng, m):
        t = m["tube_outer_diameter"]
        return {
            "inner_diameter": r2(t - rng.uniform(0.05, 0.2)),
            "wall": r2(rng.uniform(1.2, 2.5)),
            "depth": r2(rng.uniform(0.6, 1.2) * min(t, 20)),
            "top_thickness": r2(rng.uniform(1.5, 4)),
            "top_chamfer": r2(rng.uniform(0.3, 1.2)),
        }

    def derive_design(self, m, o, pr, material="PLA", bias=None):
        d = dict(o)
        from spare.features.tolerance import MATERIALS, press_allowance

        a = press_allowance("hole", m["tube_outer_diameter"], m["tube_outer_diameter"] / 2,
                            MATERIALS[material].allow_strain_snap, pr, (bias or {}).get("inner_diameter", 0.0))
        d["inner_diameter"] = r2(m["tube_outer_diameter"] + 2 * a)
        return d

    def program(self, p, style=0):
        h = header("Round end cap that pushes over a tube (top face on the bed, opening upward)", FRAME_Z)
        h += fmt_params([
            ("inner_diameter", p["inner_diameter"], "critical: press fit over tube"),
            ("wall", p["wall"], ""),
            ("depth", p["depth"], "insertion depth"),
            ("top_thickness", p["top_thickness"], ""),
            ("top_chamfer", p["top_chamfer"], "bed-side outer edge"),
        ]) + "\n"
        h += "outer_diameter = inner_diameter + 2 * wall\n"
        if style == 0:
            return h + (
                "cap = (\n    cq.Workplane(\"XY\")\n    .circle(outer_diameter / 2)\n    .extrude(top_thickness + depth)\n"
                "    .faces(\"<Z\").edges().chamfer(top_chamfer)\n)\n"
                "pocket = cq.Workplane(\"XY\").workplane(offset=top_thickness).circle(inner_diameter / 2).extrude(depth)\n"
                "result = cap.cut(pocket)\n"
            )
        return h + (
            "result = (\n    cq.Workplane(\"XY\")\n    .circle(outer_diameter / 2)\n    .extrude(top_thickness + depth)\n"
            "    .faces(\"<Z\").edges().chamfer(top_chamfer)\n    .faces(\">Z\").workplane()\n"
            "    .hole(inner_diameter, depth)\n)\n"
        )

    def mating_program(self, m, d):
        return (
            "import cadquery as cq\n"
            f"tube_outer_diameter = {m['tube_outer_diameter']}\ntube_wall = {m['tube_wall']}\n"
            f"z0 = {r2(d['top_thickness'] + 0.3)}\nlength = {r2(d['depth'] + 15)}\n"
            'result = cq.Workplane("XY").workplane(offset=z0).circle(tube_outer_diameter / 2)'
            ".circle(tube_outer_diameter / 2 - tube_wall).extrude(length)\n"
        )

    def fit_specs(self, m, d):
        z0 = d["top_thickness"] + 0.3 + 0.15 * d["depth"]
        z1 = d["top_thickness"] + 0.85 * d["depth"]
        pts, nrm = cyl_points((0, 0, 0), "z", m["tube_outer_diameter"] / 2, z0, z1, 32, 4)
        return [FitSpec("cap_on_tube", "press", "hole", pts, nrm, m["tube_outer_diameter"], "inner_diameter", "tube")]

    def probes(self, d):
        od = d["inner_diameter"] + 2 * d["wall"]
        return [
            Probe("outer_diameter", "extent", {"axis": 0}, od, 0.4, False),
            Probe("height", "extent", {"axis": 2}, d["top_thickness"] + d["depth"], 0.2, True),
            Probe("depth", "depth_from_top", {"at": [0, 0]}, d["depth"], 0.25, True),
        ]

    def strength_checks(self, m, d):
        # hoop strain from press interference: eps = interference_per_side / radius
        return [StrengthCheck("hoop_strain", "hoop_strain", {}, {
            "radius": m["tube_outer_diameter"] / 2, "fit_spec": 0})]

    def feature_graph(self, m, d, o=None):
        return {"part_type": self.part_type, "frame": FRAME_Z,
                "body": {"type": "cup", "outer_diameter": r2(d["inner_diameter"] + 2 * d["wall"]),
                         "height": r2(d["top_thickness"] + d["depth"]), "top_thickness": d["top_thickness"]},
                "features": [
                    {"id": "socket", "type": "blind_hole", "axis": "z", "opens": "+z", "diameter": d["inner_diameter"],
                     "depth": d["depth"], "critical": True, "fit": "press",
                     "mates_with": f"tube outer d={m['tube_outer_diameter']}"},
                    {"id": "edge", "type": "chamfer", "edges": "bed-side outer edge", "size": d["top_chamfer"]}]}

    def break_program(self, o, rng):
        src = self.program(o, 0)
        ang = float(rng.uniform(0, 360))
        od = o["inner_diameter"] + 2 * o["wall"]
        src += (
            f"\n# --- simulated breakage: split skirt (axial crack + missing segment) at {ang:.0f} deg\n"
            f'seg = cq.Workplane("XY").workplane(offset={o["top_thickness"] + 0.5 * o["depth"]:.2f})'
            f'.center({od / 2:.2f}, 0).rect({o["wall"] * 3:.2f}, {od * 0.35:.2f}).extrude({o["depth"]:.2f})'
            f".rotate((0, 0, 0), (0, 0, 1), {ang:.1f})\nresult = result.cut(seg)\n"
        )
        return src, {"type": "skirt_segment", "angle_deg": ang}


class WheelGrooved(Family):
    name = "wheel_grooved"
    split = "train"
    part_type = "grooved wheel / pulley on an axle"
    n_styles = 2

    def sample_mating(self, rng):
        ax = r2(float(rng.choice([3, 4, 5, 6, 8])) - rng.uniform(0, 0.05))
        return {"axle_diameter": ax}

    def sample_original(self, rng, m):
        a = m["axle_diameter"]
        od = r2(rng.uniform(max(14, 3 * a), 45))
        w = r2(rng.uniform(5, 14))
        return {
            "wheel_diameter": od,
            "width": w,
            "groove_depth": r2(rng.uniform(0.8, min(4.0, od / 8))),
            "groove_width": r2(w * rng.uniform(0.3, 0.6)),
            "hub_diameter": r2(min(od - 4, a + rng.uniform(4, 10))),
            "hub_extra": r2(rng.uniform(0, 3)),
            "bore_diameter": r2(a + rng.uniform(0.15, 0.4)),
        }

    def derive_design(self, m, o, pr, material="PLA", bias=None):
        d = dict(o)
        a = self.allow("clearance", "hole", m["axle_diameter"], pr, bias, "bore_diameter")
        d["bore_diameter"] = r2(m["axle_diameter"] + 2 * a)
        d["hub_diameter"] = max(d["hub_diameter"], r2(d["bore_diameter"] + 3))
        return d

    def program(self, p, style=0):
        h = header("Grooved wheel / pulley (axis Z, one face on the bed)", FRAME_Z)
        h += fmt_params([
            ("wheel_diameter", p["wheel_diameter"], ""),
            ("width", p["width"], "rim width"),
            ("groove_depth", p["groove_depth"], "V-groove depth"),
            ("groove_width", p["groove_width"], "V-groove opening"),
            ("hub_diameter", p["hub_diameter"], ""),
            ("hub_extra", p["hub_extra"], "hub protrusion above rim"),
            ("bore_diameter", p["bore_diameter"], "critical: clearance fit on axle"),
        ]) + "\n"
        h += "R = wheel_diameter / 2\nzc = width / 2\n"
        s = h + (
            "profile = [\n    (bore_diameter / 2, 0),\n    (R, 0),\n    (R, zc - groove_width / 2),\n"
            "    (R - groove_depth, zc),\n    (R, zc + groove_width / 2),\n    (R, width),\n"
            "    (hub_diameter / 2, width),\n    (hub_diameter / 2, width + hub_extra),\n"
            "    (bore_diameter / 2, width + hub_extra),\n]\n"
        )
        if style == 0:
            return s + 'result = cq.Workplane("XZ").polyline(profile).close().revolve(360, (0, 0, 0), (0, 1, 0))\n'
        return s + (
            'wheel = cq.Workplane("XZ").polyline(profile).close().revolve(360, (0, 0, 0), (0, 1, 0))\n'
            "result = wheel\n"
        )

    def mating_program(self, m, d):
        return ("import cadquery as cq\n"
                f"axle_diameter = {m['axle_diameter']}\nlength = {r2(d['width'] + d['hub_extra'] + 10)}\n"
                'result = cq.Workplane("XY").workplane(offset=-5).circle(axle_diameter / 2).extrude(length)\n')

    def fit_specs(self, m, d):
        H = d["width"] + d["hub_extra"]
        pts, nrm = cyl_points((0, 0, 0), "z", m["axle_diameter"] / 2, 0.15 * H, 0.85 * H, 24, 5)
        return [FitSpec("bore_on_axle", "clearance", "hole", pts, nrm, m["axle_diameter"], "bore_diameter", "axle")]

    def probes(self, d):
        return [
            Probe("wheel_diameter", "extent", {"axis": 0}, d["wheel_diameter"], 0.3, True),
            Probe("total_height", "extent", {"axis": 2}, d["width"] + d["hub_extra"], 0.2, True),
            Probe("groove_root_d", "circle_d", {"axis": 2, "offset": d["width"] / 2, "near": [0, 0], "which": "outer"},
                  d["wheel_diameter"] - 2 * d["groove_depth"], 0.3, True),
            Probe("bore_diameter", "circle_d", {"axis": 2, "offset": d["width"] / 2, "near": [0, 0], "which": "inner"},
                  d["bore_diameter"], 0.12, True),
        ]

    def feature_graph(self, m, d, o=None):
        return {"part_type": self.part_type, "frame": FRAME_Z,
                "body": {"type": "revolved_profile", "diameter": d["wheel_diameter"], "rim_width": d["width"]},
                "features": [
                    {"id": "groove", "type": "v_groove", "on": "rim", "depth": d["groove_depth"],
                     "width": d["groove_width"], "center_z": r2(d["width"] / 2), "critical": True},
                    {"id": "hub", "type": "boss", "axis": "z", "on": "top face", "diameter": d["hub_diameter"],
                     "height": d["hub_extra"]},
                    {"id": "bore", "type": "through_hole", "axis": "z", "at": [0, 0], "diameter": d["bore_diameter"],
                     "critical": True, "fit": "clearance", "mates_with": f"axle d={m['axle_diameter']}"}]}

    def break_program(self, o, rng):
        src = self.program(o, 0)
        ang = float(rng.uniform(0, 360))
        src += (f"\n# --- simulated breakage: rim chip at {ang:.0f} deg\n"
                f'chip = cq.Workplane("XY").center({o["wheel_diameter"] / 2:.2f}, 0).rect({o["wheel_diameter"] * 0.25:.2f}, '
                f'{o["wheel_diameter"] * 0.3:.2f}).extrude({o["width"] + o["hub_extra"] + 1:.2f})'
                f".rotate((0, 0, 0), (0, 0, 1), {ang:.1f})\nresult = result.cut(chip)\n")
        return src, {"type": "rim_chip", "angle_deg": ang}


_ = (np, LoadCase)
