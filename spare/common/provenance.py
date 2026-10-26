"""Provenance stamped into every results file (rule 1: never fabricate; every number is traceable)."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                               text=True, cwd=ROOT).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def data_snapshot() -> str:
    """Hash of the frozen benchmark manifest (identifies the data snapshot)."""
    m = ROOT / "data" / "benchmarks" / "MANIFEST.json"
    return file_sha256(m)[:16] if m.exists() else "none"


def stamp(seed: int | None, config: dict[str, Any], checkpoint: str | None = None,
          evidence: str = "simulated", printer: str | None = None, material: str | None = None) -> dict:
    return {
        "git_commit": git_commit(),
        "data_snapshot": data_snapshot(),
        "checkpoint": checkpoint or "none",
        "seed": seed,
        "config": config,
        "evidence_stage": evidence,
        "printer": printer,
        "material": material,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": platform.platform(),
    }


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def write_result(path: str | Path, payload: dict, provenance: dict) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"provenance": provenance, **payload}, indent=1, default=_default, allow_nan=False))
    return p


def read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


def bootstrap_ci(values, n: int = 2000, seed: int = 0, stat=np.mean, alpha: float = 0.05) -> tuple[float, float, float]:
    """Percentile bootstrap over units (the part is the unit). Returns (point, lo, hi)."""
    v = np.asarray(values, float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    bs = [stat(v[rng.integers(0, len(v), len(v))]) for _ in range(n)]
    return float(stat(v)), float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))


def paired_bootstrap_diff(a, b, n: int = 2000, seed: int = 0) -> tuple[float, float, float, float]:
    """Paired difference mean(a-b) over the same parts with CI and a two-sided bootstrap p-value."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    rng = np.random.default_rng(seed)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    return float(d.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975)), float(min(1.0, p))
