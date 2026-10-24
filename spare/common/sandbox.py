"""Sandboxed execution of CadQuery programs.

Programs (hand-written family templates or LLM output) are plain Python that must
assign the final solid to a top-level variable ``result``.  Untrusted code is
(1) AST-screened (only ``cadquery``/``math`` imports, no dunder access, no I/O
builtins) and (2) executed in a worker process with a wall-clock timeout.  The
worker returns the shape serialised as BREP text, which the parent re-imports.

This is defence-in-depth for research use, not a security boundary against a
determined adversary; DECISIONS.md records that.
"""

from __future__ import annotations

import ast
import io
import math
import multiprocessing as mp
import os
import signal
import traceback
from dataclasses import dataclass
from typing import Any

ALLOWED_IMPORTS = {"cadquery", "math"}
BANNED_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "input", "breakpoint", "exit", "quit", "help",
    "memoryview", "os", "sys", "subprocess", "shutil", "socket", "pathlib",
}
MAX_SOURCE_CHARS = 20_000


class ProgramRejected(Exception):
    pass


def screen_source(src: str) -> ast.Module:
    """Static screening of untrusted source. Raises ProgramRejected."""
    if len(src) > MAX_SOURCE_CHARS:
        raise ProgramRejected("source too long")
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        raise ProgramRejected(f"syntax error: {e.msg} (line {e.lineno})") from e
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    raise ProgramRejected(f"import not allowed: {a.name}")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                raise ProgramRejected(f"import not allowed: {node.module}")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ProgramRejected(f"dunder attribute: {node.attr}")
        elif isinstance(node, ast.Name) and (node.id in BANNED_NAMES or node.id.startswith("__")):
            raise ProgramRejected(f"banned name: {node.id}")
        elif isinstance(node, (ast.While,)):
            # while loops are a common hang source in generated code; for-loops over range are fine
            raise ProgramRejected("while loops not allowed")
    return tree


def strip_markdown(src: str) -> str:
    """Extract code from a markdown fenced block if present."""
    if "```" not in src:
        return src
    parts = src.split("```")
    best = ""
    for i in range(1, len(parts), 2):
        block = parts[i]
        if block.startswith("python"):
            block = block[len("python"):]
        if "result" in block and len(block) > len(best):
            best = block
    return best or parts[1].removeprefix("python")


def _safe_builtins() -> dict[str, Any]:
    import builtins

    ok = [
        "abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "len", "list", "max",
        "min", "pow", "range", "reversed", "round", "set", "sorted", "sum", "tuple", "zip",
        "isinstance", "print", "map", "filter", "ValueError", "Exception", "True", "False", "None",
    ]
    d = {k: getattr(builtins, k) for k in ok if hasattr(builtins, k)}

    def _imp(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
        if name.split(".")[0] not in ALLOWED_IMPORTS:
            raise ImportError(name)
        return __import__(name, globals, locals, fromlist, level)

    d["__import__"] = _imp
    return d


def exec_program_inproc(src: str, trusted: bool = False) -> Any:
    """Execute and return the cadquery object bound to ``result`` (no timeout)."""
    import cadquery as cq

    if not trusted:
        screen_source(src)
    g: dict[str, Any] = {"__builtins__": _safe_builtins(), "cq": cq, "math": math}
    exec(compile(src, "<program>", "exec"), g)  # noqa: S102
    if "result" not in g:
        raise ProgramRejected("program did not assign `result`")
    return g["result"]


def to_shape(obj: Any):
    """Convert a cadquery Workplane/Shape/Assembly-ish result to a cq.Shape (compound if needed)."""
    import cadquery as cq

    if isinstance(obj, cq.Workplane):
        vals = [v for v in obj.vals() if isinstance(v, cq.Shape)]
        if not vals:
            raise ProgramRejected("result Workplane holds no shapes")
        if len(vals) == 1:
            return vals[0]
        return cq.Compound.makeCompound(vals)
    if isinstance(obj, cq.Shape):
        return obj
    raise ProgramRejected(f"result has unsupported type {type(obj).__name__}")


def shape_to_brep(shape) -> str:
    buf = io.BytesIO()
    shape.exportBrep(buf)
    return buf.getvalue().decode("latin-1")


def brep_to_shape(s: str):
    import cadquery as cq

    return cq.Shape.importBrep(io.BytesIO(s.encode("latin-1")))


@dataclass
class ExecResult:
    ok: bool
    error: str = ""
    error_kind: str = ""  # rejected | exception | timeout | no_result
    brep: str = ""

    def shape(self):
        return brep_to_shape(self.brep) if self.ok else None


def _timeout_handler(signum, frame):
    raise TimeoutError("program timeout")


def _worker(args: tuple[str, bool, int]) -> ExecResult:
    src, trusted, timeout = args
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)
    try:
        obj = exec_program_inproc(src, trusted=trusted)
        shp = to_shape(obj)
        return ExecResult(True, brep=shape_to_brep(shp))
    except ProgramRejected as e:
        return ExecResult(False, str(e), "rejected")
    except TimeoutError:
        return ExecResult(False, "timeout", "timeout")
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exception_only(type(e), e)[-1].strip()
        return ExecResult(False, tb[:500], "exception")
    finally:
        signal.alarm(0)


def _ping(_):
    return os.getpid()


def _init_worker():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import cadquery  # noqa: F401  (warm import)


class ProgramRunner:
    """Pool of warm CadQuery workers with per-program timeout."""

    def __init__(self, workers: int | None = None, timeout: int = 20):
        self.workers = workers or max(1, (os.cpu_count() or 4) - 2)
        self.timeout = timeout
        self._pool = None

    def _ensure(self):
        if self._pool is None:
            ctx = mp.get_context("spawn")
            self._pool = ctx.Pool(self.workers, initializer=_init_worker, maxtasksperchild=200)
            # block until every worker has imported cadquery (cold OCP import can take a minute)
            self._pool.map(_ping, range(self.workers * 2), chunksize=1)

    def run_many(self, srcs: list[str], trusted: bool = False) -> list[ExecResult]:
        self._ensure()
        assert self._pool is not None
        asyncs = [self._pool.apply_async(_worker, ((s, trusted, self.timeout),)) for s in srcs]
        out: list[ExecResult] = []
        broken = False
        for a in asyncs:
            try:
                out.append(a.get(timeout=self.timeout + 15))
            except mp.TimeoutError:
                out.append(ExecResult(False, "hard timeout", "timeout"))
                broken = True
        if broken:  # a worker is stuck in C++ code; recycle the pool
            self.close()
        return out

    def run(self, src: str, trusted: bool = False) -> ExecResult:
        return self.run_many([src], trusted)[0]

    def close(self):
        if self._pool is not None:
            self._pool.terminate()
            self._pool = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def run_program_local(src: str, trusted: bool = False) -> ExecResult:
    """Run in-process (no hard timeout). Used for trusted templates and tests."""
    try:
        obj = exec_program_inproc(src, trusted=trusted)
        shp = to_shape(obj)
        return ExecResult(True, brep=shape_to_brep(shp))
    except ProgramRejected as e:
        return ExecResult(False, str(e), "rejected")
    except Exception as e:  # noqa: BLE001
        return ExecResult(False, f"{type(e).__name__}: {e}"[:500], "exception")
