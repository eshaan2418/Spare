"""The 30-second fit form (CLI). Appends a physical FitReport to data/real/<part>/fit_reports.jsonl.

  python -m spare.feedback.fit_form --part p001 --attempt 1
  python -m spare.feedback.fit_form --part p001 --attempt 1 --text "too tight on the shaft"

Interactive mode asks four questions. Raw photos stay in the git-ignored photos folder; only the relative
path is recorded.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from spare.feedback.report import OUTCOMES, FitReport, parse_text

ROOT = Path(__file__).resolve().parents[2]


def report_path(part: str, root: Path = ROOT) -> Path:
    return root / "data" / "real" / part / "fit_reports.jsonl"


def save(rep: FitReport, root: Path = ROOT) -> Path:
    p = report_path(rep.part_id, root)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps({**rep.to_dict(), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
    return p


def load(part: str, root: Path = ROOT) -> list[FitReport]:
    p = report_path(part, root)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            d.pop("timestamp", None)
            out.append(FitReport(**d))
    return out


def _ask(prompt: str, choices: tuple[str, ...] | None = None) -> str:
    while True:
        s = input(prompt).strip()
        if choices is None or s in choices:
            return s
        if s.isdigit() and choices and 1 <= int(s) <= len(choices):
            return choices[int(s) - 1]
        print("choose one of:", ", ".join(choices))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", required=True)
    ap.add_argument("--attempt", type=int, required=True)
    ap.add_argument("--text", default=None, help="free-text report, parsed automatically")
    ap.add_argument("--outcome", choices=OUTCOMES, default=None)
    ap.add_argument("--location", default="")
    ap.add_argument("--printer", default="")
    ap.add_argument("--material", default="")
    ap.add_argument("--photo", default="")
    a = ap.parse_args(argv)
    if a.text is not None:
        rep = parse_text(a.text, a.part, a.attempt)
        if a.outcome:
            rep.outcome = a.outcome
        if a.location:
            rep.location = a.location
    elif a.outcome:
        rep = FitReport(a.part, a.attempt, a.outcome, a.location)
    else:
        print("Outcome: " + "  ".join(f"{i + 1}={o}" for i, o in enumerate(OUTCOMES)))
        outcome = _ask("> ", OUTCOMES)
        loc = _ask("Where? (e.g. shaft, snap hook, screw holes; blank if n/a) > ") if outcome != "fits" else ""
        note = _ask("Anything else? > ")
        rep = FitReport(a.part, a.attempt, outcome, loc, note)
    rep.printer, rep.material, rep.photo = a.printer, a.material, a.photo
    rep.evidence = "physical"
    p = save(rep)
    print(f"saved {rep.outcome} @ '{rep.location}' -> {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
