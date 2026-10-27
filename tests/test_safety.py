import pytest

from spare.safety.safety import check, gate

REFUSED = [
    "bracket for my bunk bed frame", "step stool leg foot", "brake lever clip for my bike", "steering wheel trim clip",
    "car seat buckle cover", "baby gate latch", "crib slat connector", "inhaler cap", "CPAP hose clip",
    "gas stove knob", "propane tank cap", "pressure cooker valve", "outlet cover plate", "light switch cover",
    "230V junction box lid", "wheelchair armrest clip", "seat belt guide", "helmet buckle", "grab bar bracket",
]
WARNED = ["coffee machine drip tray clip", "bottle cap for water", "toaster knob", "outdoor garden hose clip"]
OK = ["kitchen cabinet knob on a D-shaft", "battery cover for TV remote", "shelf peg 5 mm", "cable clip for desk",
      "wire rack clip", "spacer for 4 mm bolt", "end cap for 20x20 tube", "hinge leaf for a toy box lid"]


@pytest.mark.parametrize("desc", REFUSED)
def test_refused(desc):
    r = check(desc)
    assert r.decision == "refuse", (desc, r)
    assert not gate(r, confirmed=True)


@pytest.mark.parametrize("desc", WARNED)
def test_warned_needs_confirmation(desc):
    r = check(desc)
    assert r.decision == "warn", (desc, r)
    assert r.needs_confirmation and r.messages
    assert not gate(r, confirmed=False) and gate(r, confirmed=True)


@pytest.mark.parametrize("desc", OK)
def test_ok(desc):
    assert check(desc).decision == "ok", desc


def test_numeric_heat_and_load():
    assert check("clip", max_temp_c=50, material="PLA").decision == "warn"
    assert check("clip", max_temp_c=100, material="PLA").decision == "refuse"
    assert check("clip", max_temp_c=65, material="PETG").decision == "warn"
    assert check("hook", load_n=120).decision == "warn"
    assert check("hook", load_n=10).decision == "ok"
