"""Structured representation (feature graph) -> compact text prompt for the generator."""

from __future__ import annotations

import json
from typing import Any

INSTRUCTION = (
    "Write a CadQuery (Python) program for this replacement part. Use the canonical frame given. "
    "Put every dimension in a named top-level parameter (mm) and assign the final single solid to `result`. "
    "Only `import cadquery as cq` and `import math` are allowed."
)


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.2f}".rstrip("0").rstrip(".")
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join("None" if x is None else _fmt(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}: {_fmt(x)}" for k, x in v.items()) + "}"
    return str(v)


def render(graph: dict, material: str = "PLA", printer: str = "generic_pla_0.4", sigma: dict | None = None,
           include_instruction: bool = True) -> str:
    lines = [f"PART: {graph['part_type']}", f"MATERIAL: {material}   PRINTER: {printer}", f"FRAME: {graph['frame']}"]
    body = {k: v for k, v in graph["body"].items()}
    lines.append("BODY: " + " ".join(f"{k}={_fmt(v)}" for k, v in body.items()))
    if "load" in graph:
        lines.append("LOAD: " + " ".join(f"{k}={_fmt(v)}" for k, v in graph["load"].items()))
    lines.append("FEATURES:")
    for f in graph["features"]:
        crit = f.get("critical")
        attrs = {k: v for k, v in f.items() if k not in ("id", "type", "critical", "fit", "mates_with")}
        s = f"- {f['id']}: {f['type']} " + " ".join(f"{k}={_fmt(v)}" for k, v in attrs.items())
        if sigma and f["id"] in sigma:
            s += f" (±{sigma[f['id']]:.2f})"
        if crit:
            s += f" [CRITICAL fit={f.get('fit', '-')}; mates with {f.get('mates_with', '?')}]"
        lines.append(s)
    if include_instruction:
        lines.append("")
        lines.append(INSTRUCTION)
    return "\n".join(lines)


def to_json(graph: dict) -> str:
    return json.dumps(graph, sort_keys=True)
