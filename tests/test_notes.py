import json
from types import SimpleNamespace

from powerrank import notes


def entry(team: str = "KC", rank: int = 3, change: int = 2) -> dict:
    return {
        "team": team, "name": "Kansas City Chiefs", "rank": rank, "previous_rank": rank + change,
        "rank_change": change, "tier": "A", "rating": 4.2, "rating_change": 1.1, "wins": 5, "losses": 0,
        "ties": 0, "point_diff": 57, "starter": "Patrick Mahomes", "strength_of_record_rank": 3,
        "odds": {"make_playoffs": 0.92, "win_division": 0.6, "win_super_bowl": 0.09},
        "breakdown": {"offense": 1.7, "defense": 0.6, "special_teams": -0.1, "scoring": 2.0, "quarterback": 0.0, "injuries": 0.0},
        "offseason": {"qb_changed": False, "veteran_qb": False, "coach_changed": False},
        "vegas_difference": 0.4,
    }


CTX = {
    "previous_breakdown": {"offense": 0.2, "defense": 0.5, "special_teams": 0.0, "scoring": 1.8, "quarterback": 0.0, "injuries": 0.0},
    "last_game": "a 16-6 win over Chargers (ranked #29) at home",
}


def fake_client(reply):
    """A stand-in for anthropic.Anthropic whose messages.create returns `reply`."""
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=json.dumps(reply))])

    return SimpleNamespace(messages=SimpleNamespace(create=create), calls=calls)


def test_facts_pick_out_what_moved():
    facts = notes.team_facts(entry(), CTX)
    assert facts["what_moved_since_last_week"] == {"the offense": "+1.5 points"}
    assert facts["last_game"].startswith("a 16-6 win")
    assert facts["make_playoffs_percent"] == 92
    assert "quarterback_effect" not in facts


def test_facts_flag_a_backup_quarterback():
    e = entry()
    e["breakdown"]["quarterback"] = -3.7
    assert "3.7 points below" in notes.team_facts(e, CTX)["quarterback_effect"]


def test_valid_note_requires_numbers_from_the_facts():
    facts = notes.team_facts(entry(), CTX)
    assert notes.valid_note("Up two spots to #3 after a 16-6 win, with the offense up 1.5 points.", facts)
    assert not notes.valid_note("Up to #3 after a 31-6 win.", facts)  # invented score
    assert not notes.valid_note("", facts)
    assert not notes.valid_note("word " * 80, facts)
    assert not notes.valid_note(None, facts)


def test_template_note_is_deterministic_and_passes_the_check():
    for change in (4, 1, 0, -2, -5):
        facts = notes.team_facts(entry(change=change), CTX)
        text = notes.template_note(facts)
        assert text == notes.template_note(facts)
        assert notes.valid_note(text, facts), text


def test_template_note_handles_a_bye_week():
    facts = notes.team_facts(entry(), {**CTX, "last_game": None})
    assert "after the bye week" in notes.template_note(facts)


def test_claude_notes_are_used_when_valid_and_fall_back_when_not():
    entries = [entry("KC"), entry("BUF", rank=5, change=-1)]
    contexts = {"KC": CTX, "BUF": CTX}
    client = fake_client({
        "KC": "Up two to #3 after a 16-6 win over the Chargers; the offense did the work.",
        "BUF": "Down to #5 after a 40-3 win.",  # numbers not in BUF's facts
    })
    counts = notes.attach(entries, contexts, {}, client)
    assert counts == {"reused": 0, "ai": 1, "template": 1}
    assert entries[0]["note_source"] == "ai"
    assert entries[1]["note_source"] == "template"
    assert "40-3" not in entries[1]["note"]


def test_request_asks_for_every_team_in_a_json_schema():
    entries = [entry("KC"), entry("BUF", rank=5)]
    client = fake_client({})
    notes.attach(entries, {"KC": CTX, "BUF": CTX}, {}, client)
    request = client.calls[0]
    assert request["model"] == notes.MODEL
    schema = request["output_config"]["format"]["schema"]
    assert sorted(schema["required"]) == ["BUF", "KC"]
    assert "Kansas City" in request["messages"][0]["content"]


def test_api_failure_falls_back_to_templates():
    entries = [entry()]
    counts = notes.attach(entries, {"KC": CTX}, {}, fake_client(RuntimeError("boom")))
    assert counts == {"reused": 0, "ai": 0, "template": 1}
    assert entries[0]["note_source"] == "template"


def test_no_client_means_templates_without_any_request():
    entries = [entry()]
    assert notes.attach(entries, {"KC": CTX}, {}, None)["template"] == 1


def test_unchanged_facts_reuse_the_previous_note_without_a_request():
    first = [entry()]
    notes.attach(first, {"KC": CTX}, {}, fake_client({"KC": "Up two to #3 after a 16-6 win."}))
    again = [entry()]
    client = fake_client({})
    counts = notes.attach(again, {"KC": CTX}, {"KC": first[0]}, client)
    assert counts["reused"] == 1 and not client.calls
    assert again[0]["note"] == first[0]["note"]


def test_changed_facts_get_a_new_note():
    first = [entry()]
    notes.attach(first, {"KC": CTX}, {}, None)
    changed = [entry(change=-3)]
    assert notes.attach(changed, {"KC": CTX}, {"KC": first[0]}, None)["reused"] == 0


def test_template_notes_are_replaced_once_claude_is_available():
    first = [entry()]
    notes.attach(first, {"KC": CTX}, {}, None)
    second = [entry()]
    counts = notes.attach(second, {"KC": CTX}, {"KC": first[0]}, fake_client({"KC": "Up two to #3 after a 16-6 win."}))
    assert counts["ai"] == 1
