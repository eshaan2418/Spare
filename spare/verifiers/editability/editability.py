"""Editability verifiers.

Primary (reward), static: named top-level parameters, few magic numbers in
geometry calls, descriptive names, bounded program length.
Held-out, functional: perturb each top-level numeric parameter, re-execute and
check (a) the program regenerates a valid solid and (b) every fit-critical gap
is controlled by some single named parameter (editing it moves the gap).
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

TRIVIAL = {0, 1, 2, 3, 4, 0.5, 90, 180, 360, -1, 10, 100, 1000, 0.25, 0.35, 0.3, 0.6, 0.8, 0.9, 0.01, 0.02, 0.05,
           0.1, 0.2}
MAX_LINES = 80


@dataclass
class EditResult:
    score: float
    passed: bool
    metrics: dict = field(default_factory=dict)
    backend: str = "static"


def top_level_params(tree: ast.Module) -> dict[str, float]:
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            v = node.value
            if isinstance(v, ast.Constant) and isinstance(v.value, (int, float)) and not isinstance(v.value, bool):
                out[node.targets[0].id] = float(v.value)
            elif isinstance(v, ast.UnaryOp) and isinstance(v.op, ast.USub) and isinstance(v.operand, ast.Constant):
                out[node.targets[0].id] = -float(v.operand.value)  # type: ignore[arg-type]
    return out


def static_editability(src: str) -> EditResult:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return EditResult(0.0, False, {"error": "syntax"})
    params = top_level_params(tree)
    param_nodes = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in params:
            param_nodes.add(id(node.value))
            if isinstance(node.value, ast.UnaryOp):
                param_nodes.add(id(node.value.operand))
    magic = 0
    lits = 0
    for sub in ast.walk(tree):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, (int, float)) and not isinstance(sub.value, bool):
            if id(sub) in param_nodes:
                continue
            lits += 1
            if abs(sub.value) not in TRIVIAL:
                magic += 1
    names = list(params)
    good_names = [n for n in names if len(n) >= 3 and re.fullmatch(r"[a-z][a-z0-9_]*", n) and not re.fullmatch(r"[a-z]\d*", n)]
    lines = [ln for ln in src.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    n_lines = len(lines)
    s_params = min(len(params) / 3.0, 1.0)
    s_magic = 1.0 / (1.0 + 0.25 * magic)
    s_names = len(good_names) / max(len(names), 1) if names else 0.0
    s_len = 1.0 if n_lines <= MAX_LINES else max(0.0, 1 - (n_lines - MAX_LINES) / MAX_LINES)
    has_comments = "#" in src
    score = 0.3 * s_params + 0.3 * s_magic + 0.2 * s_names + 0.15 * s_len + 0.05 * float(has_comments)
    return EditResult(float(score), bool(score >= 0.7), {"n_params": len(params), "magic_numbers": magic,
                                                         "literals": lits, "good_name_frac": round(s_names, 3),
                                                         "lines": n_lines}, "static")


def _set_param(src: str, name: str, value: float) -> str:
    return re.sub(rf"^({re.escape(name)}\s*=\s*)[-+0-9.eE]+", lambda m: f"{m.group(1)}{value:.4f}", src, count=1,
                  flags=re.M)


def functional_editability(src: str, gap_fn, run_fn, delta: float = 0.2, max_params: int = 16) -> EditResult:
    """gap_fn(shape) -> list of median design gaps per fit spec (np.nan on failure).
    run_fn(src) -> shape or None."""
    import numpy as np

    try:
        params = top_level_params(ast.parse(src))
    except SyntaxError:
        return EditResult(0.0, False, {"error": "syntax"}, "functional")
    base_shape = run_fn(src)
    if base_shape is None:
        return EditResult(0.0, False, {"error": "base program invalid"}, "functional")
    g0 = np.asarray(gap_fn(base_shape), float)
    controlled = np.zeros(len(g0), bool)
    ok = 0
    tried = 0
    for name, v in list(params.items())[:max_params]:
        if float(v).is_integer() and abs(v) < 64 and ("count" in name or name.endswith("s") or name.startswith("n_")):
            continue  # integer counts: skip
        tried += 1
        shp = run_fn(_set_param(src, name, v + delta * (1 if v >= 0 else -1)))
        if shp is None:
            continue
        ok += 1
        g1 = np.asarray(gap_fn(shp), float)
        with np.errstate(invalid="ignore"):
            controlled |= np.nan_to_num(np.abs(g1 - g0), nan=0.0) >= 0.25 * delta
    regen = ok / max(tried, 1)
    ctrl = float(controlled.mean()) if len(controlled) else 1.0
    score = 0.5 * regen + 0.5 * ctrl
    return EditResult(float(score), bool(regen >= 0.8 and ctrl >= 0.99), {"regen_stability": round(regen, 3),
                                                                          "fit_dims_controllable": round(ctrl, 3),
                                                                          "params_tried": tried}, "functional")
