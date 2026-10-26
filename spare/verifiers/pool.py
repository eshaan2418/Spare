"""Parallel verification: a spawn pool of warm workers that each run ``verify`` on (program, record) jobs.

Each job has a hard wall-clock timeout; a stuck worker (OCCT can hang in C++) recycles the pool and the
job is reported as a timeout with zero reward.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
from dataclasses import dataclass, field


@dataclass
class VerifyJob:
    src: str
    rec: dict
    suite: str = "primary"
    printer: str | None = None  # None -> default printer; else a VIRTUAL_PRINTERS key
    weights: dict | None = None
    opts: dict = field(default_factory=dict)  # slicer/fea/functional_edit flags


def _timeout(signum, frame):
    raise TimeoutError("verify timeout")


def _run(args: tuple[VerifyJob, int]) -> dict:
    job, timeout = args
    from spare.verifiers.suite import printer_by_name, task_from_record, verify

    signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(timeout)
    try:
        t = task_from_record(job.rec, printer_by_name(job.printer))
        return verify(job.src, t, suite=job.suite, weights=job.weights, **job.opts).to_dict()
    except TimeoutError:
        return {"ok_exec": False, "error": "timeout", "error_kind": "timeout", "reward": 0.0, "all_pass": False,
                "suite": job.suite}
    except Exception as e:  # noqa: BLE001
        return {"ok_exec": False, "error": f"{type(e).__name__}: {e}"[:300], "error_kind": "verifier_error",
                "reward": 0.0, "all_pass": False, "suite": job.suite}
    finally:
        signal.alarm(0)


def _init():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import cadquery  # noqa: F401


def _ping(_):
    return os.getpid()


class VerifyPool:
    def __init__(self, workers: int | None = None, timeout: int = 60):
        self.workers = workers or max(1, (os.cpu_count() or 4) - 2)
        self.timeout = timeout
        self._pool = None

    def _ensure(self):
        if self._pool is None:
            self._pool = mp.get_context("spawn").Pool(self.workers, initializer=_init, maxtasksperchild=100)
            self._pool.map(_ping, range(self.workers * 2), chunksize=1)

    def run(self, jobs: list[VerifyJob]) -> list[dict]:
        self._ensure()
        assert self._pool is not None
        asyncs = [self._pool.apply_async(_run, ((j, self.timeout),)) for j in jobs]
        out, broken = [], False
        for j, a in zip(jobs, asyncs):
            try:
                out.append(a.get(timeout=self.timeout + 30))
            except mp.TimeoutError:
                out.append({"ok_exec": False, "error": "hard timeout", "error_kind": "timeout", "reward": 0.0,
                            "all_pass": False, "suite": j.suite})
                broken = True
        if broken:
            self.close()
        return out

    def close(self):
        if self._pool is not None:
            self._pool.terminate()
            self._pool = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
