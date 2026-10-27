"""Structured print-test fit reports (the 30-second fit form) and a free-text parser."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

OUTCOMES = ("fits", "too_tight", "too_loose", "broke", "did_not_print")


@dataclass
class FitReport:
    part_id: str
    attempt: int
    outcome: str  # one of OUTCOMES
    location: str = ""  # e.g. "shaft", "snap hook", "screw holes"
    note: str = ""
    photo: str = ""  # optional relative path (never committed if it is a raw photo)
    printer: str = ""
    material: str = ""
    evidence: str = "physical"  # physical (human) | simulated (verifier-generated)

    def to_dict(self) -> dict:
        return asdict(self)


_PAT = [
    ("broke", r"\b(broke|broken|snapped|cracked|split|sheared|failed)\b"),
    ("too_tight", r"\b(too\s+tight|tight|won'?t\s+fit|doesn'?t\s+fit|does not fit|too\s+small|interferes|"
                  r"can'?t\s+(push|insert)|binds?)\b"),
    ("too_loose", r"\b(too\s+loose|loose|wobbl\w*|falls?\s+(off|out)|slips?|too\s+big|rattles?)\b"),
    ("did_not_print", r"\b(spaghetti|failed\s+print|did\s*n.?t\s+print|warped\s+off)\b"),
    ("fits", r"\b(fits|works|perfect|good\s+fit|clicks?\s+in|snug)\b"),
]


def parse_text(text: str, part_id: str = "", attempt: int = 1) -> FitReport:
    """'too tight on shaft' -> FitReport(outcome=too_tight, location=shaft)."""
    t = text.strip().lower()
    outcome = "fits"
    for name, pat in _PAT:
        if re.search(pat, t):
            outcome = name
            break
    loc = ""
    m = re.search(r"\b(?:on|at|in|around|near|of)\s+(?:the\s+)?([a-z][a-z \-]{1,30}?)(?:[.,;!]|$|\s+(?:when|but|and))", t)
    if m:
        loc = m.group(1).strip()
    return FitReport(part_id, attempt, outcome, loc, note=text.strip())
