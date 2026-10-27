"""Safety screen for replacement-part requests.

Two outcomes besides ``ok``:
  * ``refuse`` -- safety-critical categories (people-load, vehicle control, child safety, medical, gas/fuel,
    pressure, mains electrical). The pipeline will not generate a part.
  * ``warn``   -- allowed only after explicit user confirmation (food contact, heat near/above the material
    rating, outdoor/UV, sustained load above a threshold).

The check is a transparent keyword/phrase classifier over the part description, user hints and the stated
use, plus numeric checks (temperature vs material rating, load). It is deliberately conservative: a false
refusal costs the user a manual workaround; a false accept can hurt someone. Keyword screens are
bypassable by a determined user and are not a substitute for judgement -- this is stated in the app.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from spare.features.tolerance import MATERIALS

REFUSE: dict[str, list[str]] = {
    "people_load": [r"\bladder", r"\bstep ?stool", r"\bchair\b", r"\bseat\b(?! ?belt)", r"\bstool\b", r"\bharness",
                    r"\bclimb", r"\bbed (frame|slat)", r"\bbunk", r"\bgrab ?bar", r"\bhandrail", r"\bbalcon",
                    r"\bswing\b", r"\bhammock", r"\bweight[- ]bearing", r"\bsupports? (a )?(person|people|body)",
                    r"\bcarabiner", r"\bbelay", r"\brope anchor", r"\bscaffold"],
    "vehicle_control": [r"\bbrake", r"\bsteering", r"\bthrottle", r"\bclutch", r"\bseat ?belt", r"\bairbag",
                        r"\bwheel (nut|bearing|hub)", r"\bsuspension", r"\baxle (nut|bearing)", r"\bbike (stem|fork)",
                        r"\bhandlebar", r"\bpedal crank", r"\bhelmet", r"\btow ?(bar|hitch)"],
    "child_safety": [r"\bcar ?seat", r"\bcrib", r"\bcot\b", r"\bbaby gate", r"\bsafety gate", r"\bstair gate",
                     r"\bhigh ?chair", r"\bstroller", r"\bpram", r"\bchild ?(lock|proof|safety)", r"\bpacifier",
                     r"\bteether", r"\btoy for (a )?(baby|infant|toddler)", r"\bsmall parts? .*(baby|infant)"],
    "medical": [r"\bmedical", r"\binhaler", r"\bcpap", r"\bnebuli[sz]er", r"\binsulin", r"\bsyringe", r"\bcatheter",
                r"\bhearing aid", r"\bprosthe", r"\borthotic", r"\bwheelchair", r"\bwalker\b", r"\bcrutch",
                r"\bsplint", r"\boxygen", r"\bpacemaker", r"\bimplant", r"\bsurgical", r"\bdental"],
    "gas_fuel": [r"\bgas (valve|line|cap|regulator|knob|hob|stove|tank|bottle|cylinder)", r"\bpropane", r"\bbutane",
                 r"\bfuel", r"\bpetrol", r"\bgasoline", r"\bdiesel", r"\bkerosene", r"\bfurnace", r"\bboiler",
                 r"\bwater heater"],
    "pressure": [r"\bpressure (cooker|vessel|tank|relief|regulator|washer)", r"\bcompressed air", r"\bair ?compressor",
                 r"\bco2 (canister|cartridge)", r"\bsoda ?stream", r"\bpneumatic", r"\bhydraulic", r"\bscuba",
                 r"\bfire extinguisher", r"\bpsi\b"],
    "mains_electrical": [r"\bmains", r"\boutlet cover", r"\bsocket (cover|plate|box)", r"\bwall (plate|socket)",
                         r"\bplug (housing|cover|body)", r"\bswitch (plate|cover)", r"\bjunction box",
                         r"\bfuse ?(box|holder)", r"\bbreaker", r"\b(110|120|220|230|240) ?v", r"\bpower supply (case|enclosure|housing)",
                         r"\blight fixture", r"\blamp (socket|holder)", r"\bextension cord", r"\bpower strip"],
}

WARN: dict[str, list[str]] = {
    "food_contact": [r"\bfood", r"\bcup\b", r"\bmug", r"\bplate\b(?! ?(holder|rack))", r"\bbowl", r"\butensil",
                     r"\bspoon", r"\bfork\b", r"\bknife", r"\bcutting board", r"\bbottle (cap|lid)", r"\bdrink",
                     r"\bkettle", r"\bblender", r"\bstraw\b", r"\bcoffee", r"\btea\b", r"\blunch ?box", r"\bmouth"],
    "heat": [r"\boven", r"\bstove", r"\bhob\b", r"\btoaster", r"\bkettle", r"\bhair ?dryer", r"\bcurling iron",
             r"\bradiator", r"\bheater", r"\bdishwasher", r"\bcar (dash|interior)", r"\bengine", r"\bexhaust",
             r"\bsoldering", r"\bfireplace", r"\bgrill", r"\bbbq", r"\bsauna"],
    "outdoor_uv": [r"\boutdoor", r"\bgarden", r"\bsun\b", r"\buv\b", r"\bpatio"],
}

LOAD_WARN_N = 50.0  # sustained loads above this need a strength check by the user
REFUSE_REASON = {
    "people_load": "it would carry a person's weight",
    "vehicle_control": "it is part of a vehicle's control or safety system",
    "child_safety": "it is a child-safety item",
    "medical": "it is part of a medical device",
    "gas_fuel": "it is part of a gas or fuel system",
    "pressure": "it is part of a pressurised system",
    "mains_electrical": "it is part of a mains-electrical enclosure or fitting",
}
WARN_TEXT = {
    "food_contact": "FDM prints have crevices that trap bacteria and most filaments are not certified food-safe. "
                    "Use food-safe filament, a food-safe coating, and avoid direct contact with food or mouths.",
    "heat": "The part may be near a heat source. Check the temperature against the material rating; PLA softens "
            "around 55 C and PETG around 70 C.",
    "outdoor_uv": "PLA degrades and creeps outdoors. Prefer PETG or ASA.",
    "load": "The stated load is high for a printed plastic part. Use a safety factor and do not hang anything over "
            "people.",
}


@dataclass
class SafetyResult:
    decision: str  # ok | warn | refuse
    categories: list[str] = field(default_factory=list)
    matches: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    needs_confirmation: bool = False

    def to_dict(self) -> dict:
        return {"decision": self.decision, "categories": self.categories, "matches": self.matches,
                "messages": self.messages, "needs_confirmation": self.needs_confirmation}


def _scan(text: str, table: dict[str, list[str]]) -> tuple[list[str], list[str]]:
    cats, hits = [], []
    for cat, pats in table.items():
        for p in pats:
            m = re.search(p, text)
            if m:
                cats.append(cat)
                hits.append(m.group(0).strip())
                break
    return cats, hits


def check(description: str, use: str = "", material: str = "PLA", max_temp_c: float | None = None,
          load_n: float | None = None) -> SafetyResult:
    text = f" {description} {use} ".lower()
    cats, hits = _scan(text, REFUSE)
    if cats:
        msgs = [f"Spare will not make this part because {REFUSE_REASON[c]}. A failed print could injure someone; "
                f"buy a certified replacement from the manufacturer." for c in cats]
        return SafetyResult("refuse", cats, hits, msgs)
    wcats, whits = _scan(text, WARN)
    mat = MATERIALS.get(material.upper())
    if max_temp_c is not None and mat is not None and max_temp_c > mat.max_temp_c - 10:
        if max_temp_c > mat.max_temp_c + 20:
            return SafetyResult("refuse", ["heat"], [f"{max_temp_c} C"],
                                [f"{max_temp_c} C is far above {mat.name}'s rating ({mat.max_temp_c} C)."])
        if "heat" not in wcats:
            wcats.append("heat")
            whits.append(f"{max_temp_c} C")
    if load_n is not None and load_n > LOAD_WARN_N:
        wcats.append("load")
        whits.append(f"{load_n} N")
    if wcats:
        return SafetyResult("warn", wcats, whits, [WARN_TEXT[c] for c in wcats], needs_confirmation=True)
    return SafetyResult("ok")


def gate(res: SafetyResult, confirmed: bool) -> bool:
    """True if generation may proceed."""
    if res.decision == "refuse":
        return False
    if res.decision == "warn":
        return confirmed
    return True
