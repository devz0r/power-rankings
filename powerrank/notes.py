"""A short note for each team explaining its week.

The code works out the facts (why the rating moved, last game, QB and
injury situation, odds, Vegas gap). Claude turns them into one or two natural
sentences, seeing all 32 teams at once so the notes don't repeat each other.
Every number in a note must appear in that team's facts, so the model can
phrase things but cannot invent results; anything that fails the check, or
any API trouble, falls back to a plain template. Without an API key every
note is a template.

A note is reused until its facts change, so a rerun with nothing new
neither changes the page nor costs anything.
"""

from __future__ import annotations

import hashlib
import json
import os
import re

MODEL = os.environ.get("POWERRANK_NOTES_MODEL", "claude-sonnet-5-5")
MAX_WORDS = 60
MAX_CHARS = 360

UNIT_LABELS = {
    "offense": "the offense",
    "defense": "the defense",
    "special_teams": "special teams",
    "scoring": "scoring margin",
    "quarterback": "the quarterback situation",
    "injuries": "injuries",
}
CORE_UNITS = ("offense", "defense", "special_teams")

SYSTEM_PROMPT = """You write the one-to-two sentence blurb under each team in a weekly NFL power ranking. The reader is a football fan who wants to know why the team is where it is and what changed this week.

You get a JSON object of verified facts per team. Use only those facts. Do not add injuries, streaks, matchups, names or storylines that aren't in them, and don't speculate about causes beyond what they say.

Voice: a knowledgeable friend, plain and specific, not a press release. Lead with whatever is most interesting about that team this week, which is not always the rank change. Vary sentence shapes and openings across the 32 notes; if two notes would start alike, change one. Avoid cliches ("rolls on", "statement win", "keep an eye on"), exclamation marks, emoji and hedging filler.

Numbers: every number you write must appear in that team's facts, copied the same way (rank, scores, points, percentages). Do not do arithmetic or round differently. Rating points are "points" better or worse than an average team. Keep each note under 45 words."""


def _facts_text(facts: dict) -> str:
    return json.dumps(facts, sort_keys=True, ensure_ascii=False)


def facts_hash(facts: dict) -> str:
    return hashlib.sha1(_facts_text(facts).encode()).hexdigest()[:16]


def _pts(v: float) -> str:
    return f"{abs(v):.1f}"


def _signed(v: float) -> str:
    return f"{'+' if v >= 0 else '-'}{abs(v):.1f}"


def team_facts(entry: dict, ctx: dict) -> dict:
    """Plain, pre-formatted facts for one team. Numbers are stored the way a
    note should print them so the model copies instead of computing."""
    deltas = {
        unit: round(entry["breakdown"][unit] - ctx["previous_breakdown"].get(unit, 0.0), 1)
        for unit in UNIT_LABELS
    }
    moved = sorted((u for u in deltas if abs(deltas[u]) >= 0.5), key=lambda u: -abs(deltas[u]))
    core = {u: entry["breakdown"][u] for u in CORE_UNITS}
    odds = entry["odds"]
    facts: dict = {
        "team": entry["name"],
        "rank": entry["rank"],
        "previous_rank": entry["previous_rank"],
        "rank_change": entry["rank_change"],
        "tier": entry["tier"],
        "rating_points": _signed(entry["rating"]),
        "rating_change_points": _signed(entry["rating_change"]),
        "record": f"{entry['wins']}-{entry['losses']}" + (f"-{entry['ties']}" if entry["ties"] else ""),
        "point_differential": entry["point_diff"],
        "quarterback": entry["starter"],
        "make_playoffs_percent": round(odds["make_playoffs"] * 100),
        "win_division_percent": round(odds["win_division"] * 100),
        "win_super_bowl_percent": round(odds["win_super_bowl"] * 100),
        "strength_of_record_rank": entry["strength_of_record_rank"],
        "what_moved_since_last_week": {
            UNIT_LABELS[u]: _signed(deltas[u]) + " points" for u in moved[:3]
        },
        "best_unit": f"{UNIT_LABELS[max(core, key=core.get)]} ({_signed(max(core.values()))} points)",
        "weakest_unit": f"{UNIT_LABELS[min(core, key=core.get)]} ({_signed(min(core.values()))} points)",
    }
    if ctx.get("last_game"):
        facts["last_game"] = ctx["last_game"]
    else:
        facts["last_game"] = "bye week"
    if ctx.get("streak"):
        facts["streak"] = ctx["streak"]
    if ctx.get("next_game"):
        facts["next_game"] = ctx["next_game"]
    qb_points = entry["breakdown"]["quarterback"]
    if abs(qb_points) >= 1.0:
        facts["quarterback_effect"] = (
            f"this week's starter puts them {_pts(qb_points)} points "
            f"{'above' if qb_points > 0 else 'below'} their usual quarterback situation"
        )
    inj = entry["breakdown"]["injuries"]
    if abs(inj) >= 0.5:
        facts["injury_effect"] = f"players ruled out cost them {_pts(inj)} points this week"
    if entry["offseason"]["qb_changed"] or entry["offseason"]["coach_changed"]:
        facts["offseason_changes"] = [
            c for c, on in (("new quarterback", entry["offseason"]["qb_changed"]), ("new head coach", entry["offseason"]["coach_changed"])) if on
        ]
    if entry.get("vegas_difference") is not None and abs(entry["vegas_difference"]) >= 2.0:
        d = entry["vegas_difference"]
        facts["vs_vegas"] = f"the model rates them {_pts(d)} points {'higher' if d > 0 else 'lower'} than the betting market does"
    return facts


# --- Template fallback ----------------------------------------------------------


def _pick(facts: dict, options: list[str], salt: str) -> str:
    h = int(hashlib.sha1((facts["team"] + salt).encode()).hexdigest(), 16)
    return options[h % len(options)]


def template_note(facts: dict) -> str:
    n = facts["rank_change"]
    rank = facts["rank"]
    if n >= 3:
        move = f"Climbs {n} spots to #{rank}"
    elif n >= 1:
        move = f"Up {n} to #{rank}"
    elif n == 0:
        move = f"Holds at #{rank}"
    elif n >= -2:
        move = f"Slips {-n} to #{rank}"
    else:
        move = f"Falls {-n} spots to #{rank}"
    last = facts["last_game"]
    after = " after the bye week" if last == "bye week" else f" after {last}"
    first = move + after + "."

    moved = list(facts["what_moved_since_last_week"].items())
    if moved:
        unit, change = moved[0]
        second = _pick(
            facts,
            [f"The biggest swing was {unit}, {change}.", f"Mostly {unit}: {change}."],
            "move",
        )
    else:
        second = "Not much changed under the hood this week."

    extra = facts.get("quarterback_effect") or facts.get("injury_effect") or facts.get("vs_vegas") or ""
    third = (extra[0].upper() + extra[1:] + ".") if extra else ""
    return " ".join(part for part in (first, second, third) if part)


# --- Claude -----------------------------------------------------------------------


def make_client():
    """An Anthropic client, or None if there is no key or the SDK is missing."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
    except ImportError:
        return None
    return anthropic.Anthropic()


def valid_note(note: object, facts: dict) -> bool:
    """A note is acceptable if it is short and every number in it comes from the facts."""
    if not isinstance(note, str) or not note.strip():
        return False
    if len(note) > MAX_CHARS or len(note.split()) > MAX_WORDS:
        return False
    allowed = {float(x) for x in re.findall(r"\d+(?:\.\d+)?", _facts_text(facts))}
    return all(float(x) in allowed for x in re.findall(r"\d+(?:\.\d+)?", note))


def ask_claude(client, all_facts: dict[str, dict]) -> dict[str, str]:
    """One request for every team; returns {team abbreviation: note} (possibly partial)."""
    schema = {
        "type": "object",
        "properties": {abbr: {"type": "string"} for abbr in all_facts},
        "required": list(all_facts),
        "additionalProperties": False,
    }
    response = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}},
        messages=[{
            "role": "user",
            "content": "Facts for each team, keyed by abbreviation:\n\n" + json.dumps(all_facts, indent=1, ensure_ascii=False),
        }],
    )
    if response.stop_reason != "end_turn":
        raise RuntimeError(f"stopped with {response.stop_reason}")
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


def attach(entries: list[dict], contexts: dict[str, dict], previous: dict[str, dict], client=None) -> dict[str, int]:
    """Add `note`, `note_source` ("ai" or "template") and `note_hash` to each
    entry. `previous` maps team -> an earlier entry carrying those fields.
    Returns counts of reused, ai and template notes."""
    all_facts = {e["team"]: team_facts(e, contexts[e["team"]]) for e in entries}
    hashes = {t: facts_hash(f) for t, f in all_facts.items()}
    counts = {"reused": 0, "ai": 0, "template": 0}

    reuse: dict[str, dict] = {}
    for t, prev in previous.items():
        if t in hashes and prev.get("note_hash") == hashes[t] and prev.get("note"):
            # A template note is worth replacing once Claude is available.
            if prev.get("note_source") == "ai" or client is None:
                reuse[t] = prev

    fresh: dict[str, str] = {}
    todo = {t: f for t, f in all_facts.items() if t not in reuse}
    if todo and client is not None:
        try:
            fresh = ask_claude(client, todo)
        except Exception as exc:  # any API or parsing trouble falls back to templates
            print(f"notes: Claude unavailable ({type(exc).__name__}: {exc}); using templates")

    for e in entries:
        t = e["team"]
        e["note_hash"] = hashes[t]
        if t in reuse:
            e["note"], e["note_source"] = reuse[t]["note"], reuse[t]["note_source"]
            counts["reused"] += 1
        elif valid_note(fresh.get(t), all_facts[t]):
            e["note"], e["note_source"] = fresh[t].strip(), "ai"
            counts["ai"] += 1
        else:
            e["note"], e["note_source"] = template_note(all_facts[t]), "template"
            counts["template"] += 1
    return counts
