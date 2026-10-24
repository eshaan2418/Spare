"""Registry of procedural part families and their frozen split assignment.

Splits are *by family*: test families are never seen in SFT/RL data, compensation
fitting, or prompt few-shots (see DECISIONS.md D-006).
"""

from __future__ import annotations

from spare.synth.families.prismatic import (
    BatteryCover,
    CableClip,
    EndCapRect,
    HingeLeaf,
    LBracket,
    RackClip,
    ShelfPeg,
    WallHook,
)
from spare.synth.families.rotational import CapRound, KnobDShaft, Spacer, WheelGrooved
from spare.synth.family import Family

FAMILIES: dict[str, Family] = {f.name: f for f in [
    Spacer(), KnobDShaft(), CapRound(), WheelGrooved(), BatteryCover(), LBracket(), CableClip(), ShelfPeg(),
    WallHook(), HingeLeaf(), RackClip(), EndCapRect(),
]}


def families_in(split: str) -> list[Family]:
    return [f for f in FAMILIES.values() if f.split == split]


def get(name: str) -> Family:
    return FAMILIES[name]
