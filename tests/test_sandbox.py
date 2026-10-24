import pytest

from spare.common.sandbox import (
    ProgramRejected,
    ProgramRunner,
    run_program_local,
    screen_source,
    strip_markdown,
)

GOOD = "import cadquery as cq\nlength = 10.0\nresult = cq.Workplane('XY').box(length, 5, 2)\n"


def test_runs_good_program():
    r = run_program_local(GOOD)
    assert r.ok, r.error
    assert abs(r.shape().Volume() - 100.0) < 1e-6


@pytest.mark.parametrize("src", [
    "import os\nresult = 1",
    "import cadquery as cq\nresult = cq.__dict__",
    "open('x','w')\nresult = 1",
    "while True:\n    pass\nresult = 1",
    "from subprocess import run\nresult = 1",
    "x = (1).__class__\nresult = x",
])
def test_rejects_unsafe(src):
    with pytest.raises(ProgramRejected):
        screen_source(src)
    assert run_program_local(src).error_kind == "rejected"


def test_missing_result():
    r = run_program_local("import cadquery as cq\nx = cq.Workplane().box(1,1,1)\n")
    assert not r.ok and r.error_kind == "rejected"


def test_exception_reported():
    r = run_program_local("import cadquery as cq\nresult = cq.Workplane().box(1,1,1).fillet(5)\n")
    assert not r.ok and r.error_kind == "exception"


def test_strip_markdown():
    assert "result" in strip_markdown("Here:\n```python\n" + GOOD + "```\nDone")


def test_pool_runner_and_timeout():
    slow = "import cadquery as cq\nn = 0\nfor i in range(10**9):\n    n += i\nresult = cq.Workplane().box(1,1,1)\n"
    with ProgramRunner(workers=2, timeout=2) as pr:
        out = pr.run_many([GOOD, slow])
    assert out[0].ok
    assert not out[1].ok and out[1].error_kind == "timeout"
