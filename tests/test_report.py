import itertools

import pytest

from powerrank import market, resume, tiers, uncertainty
from powerrank.data import Game
from powerrank.simulate import simulate
from powerrank.teams import Team


def game(i: int, home: str, away: str, week: int = 1, margin: int | None = None, spread: float | None = None) -> Game:
    played = margin is not None
    return Game(
        game_id=f"g{i}", season=2026, week=week, game_type="REG", home=home, away=away, neutral=False,
        gameday=None, home_score=(20 + max(margin, 0)) if played else None,
        away_score=(20 + max(-margin, 0)) if played else None, spread_line=spread,
    )


def test_natural_breaks_split_at_the_gaps():
    ratings = {"a": 9.0, "b": 8.5, "c": 3.0, "d": 2.8, "e": 2.5, "f": -6.0, "g": -6.5}
    labels = tiers.natural_breaks(list(ratings.values()), 3)
    assert labels == [0, 0, 1, 1, 1, 2, 2]


def test_tiers_use_letters_from_the_top():
    ratings = {f"t{i}": float(i * i) for i in range(10)}
    assigned = tiers.assign(ratings)
    assert assigned["t9"] == "S"
    assert assigned["t0"] == "F"


def faced(games, ratings):
    """Every team at its listed rating in every game."""
    return {(g.game_id, t): ratings[t] for g in games for t in (g.home, g.away)}


def test_strength_of_record_rewards_tougher_wins():
    ratings = {"us": 0.0, "strong": 6.0, "weak": -6.0}
    vs_strong = [game(i, "us", "strong", week=i + 1, margin=3) for i in range(3)]
    vs_weak = [game(i, "us", "weak", week=i + 1, margin=3) for i in range(3)]
    tough = resume.strength_of_record("us", vs_strong, faced(vs_strong, ratings), hfa=0.0, reference=2.0)
    easy = resume.strength_of_record("us", vs_weak, faced(vs_weak, ratings), hfa=0.0, reference=2.0)
    assert tough < easy  # a typical playoff team would rarely go 3-0 against strong teams


def test_strength_of_record_coin_flip():
    ratings = {"us": 0.0, "them": 0.0}
    won, lost = [game(0, "us", "them", margin=1)], [game(0, "us", "them", margin=-1)]
    assert resume.strength_of_record("us", won, faced(won, ratings), 0.0, 0.0) == pytest.approx(0.5)
    assert resume.strength_of_record("us", lost, faced(lost, ratings), 0.0, 0.0) == pytest.approx(1.0)


def test_notable_results():
    ratings = {"us": 0.0, "good": 5.0, "bad": -5.0, "meh": 0.0}
    played = [game(0, "us", "good", margin=7), game(1, "us", "bad", week=2, margin=-3), game(2, "meh", "us", week=3, margin=-10)]
    notable = resume.notable_results("us", played, faced(played, ratings))
    assert notable["best_win"]["opponent"] == "good"
    assert notable["worst_loss"]["opponent"] == "bad"


def test_opponents_are_judged_as_fielded_that_day():
    # Same opponent twice: once with its starter (+5), once with a backup (-3).
    played = [game(0, "us", "them", margin=3), game(1, "us", "them", week=2, margin=3)]
    strength = {("g0", "them"): 5.0, ("g1", "them"): -3.0, ("g0", "us"): 0.0, ("g1", "us"): 0.0}
    best = resume.notable_results("us", played, strength)["best_win"]
    assert best["week"] == 1 and best["opponent_rating"] == 5.0
    # Beating the full-strength version is harder for a reference team to match.
    only_starter = resume.strength_of_record("us", played[:1], strength, 0.0, 0.0)
    only_backup = resume.strength_of_record("us", played[1:], strength, 0.0, 0.0)
    assert only_starter < only_backup


def test_market_ratings_recover_spreads():
    truth = {"a": 6.0, "b": 2.0, "c": -2.0, "d": -6.0}
    games = [
        game(i, h, a, week=1 + i // 2, spread=truth[h] - truth[a] + 1.5)
        for i, (h, a) in enumerate(itertools.permutations(truth, 2))
    ]
    ratings, hfa = market.implied_ratings(games, latest_week=6)
    assert hfa == pytest.approx(1.5, abs=0.05)
    for t, r in truth.items():
        assert ratings[t] == pytest.approx(r, rel=0.03)  # the token prior shrinks ~2% here


def test_rating_sd_lookup():
    calibration = {"game_sd": 13.5, "rating_sd": [[1, 1, 4.3], [2, 6, 3.9], [7, 99, 3.7]]}
    assert uncertainty.rating_sd(calibration, 1) == 4.3
    assert uncertainty.rating_sd(calibration, 4) == 3.9
    assert uncertainty.rating_sd(calibration, 30) == 3.7


def league() -> dict[str, Team]:
    teams = {}
    for conf in ("AFC", "NFC"):
        for d in ("East", "North", "South", "West"):
            for k in range(4):
                abbr = f"{conf[0]}{d[0]}{k}"
                teams[abbr] = Team(abbr, abbr, abbr, conf, f"{conf} {d}", "#000", "#fff", "")
    return teams


def test_simulation_odds_add_up():
    teams = league()
    abbrs = sorted(teams)
    remaining = [(game(i, abbrs[i % 32], abbrs[(i * 7 + 3) % 32], week=1 + i // 16), 0.0) for i in range(64)]
    remaining = [(g, m) for g, m in remaining if g.home != g.away]
    odds = simulate(teams, [], remaining, dict.fromkeys(teams, 0.0), 3.0, 13.5, 2.0, n_sims=500)
    assert sum(odds.playoffs.values()) == pytest.approx(14)
    assert sum(odds.division.values()) == pytest.approx(8)
    assert sum(odds.bye.values()) == pytest.approx(2)
    assert sum(odds.conference.values()) == pytest.approx(2)
    assert sum(odds.super_bowl.values()) == pytest.approx(1)


def test_simulation_respects_played_results():
    teams = league()
    abbrs = sorted(teams)
    # AE0 beats everyone in its conference; nothing left to play.
    afc = [t for t in abbrs if teams[t].conference == "AFC" and t != "AE0"]
    played = [game(i, "AE0", t, week=i + 1, margin=10) for i, t in enumerate(afc)]
    odds = simulate(teams, played, [], dict.fromkeys(teams, 0.0), 3.0, 13.5, 2.0, n_sims=200)
    assert odds.division["AE0"] == 1.0
    assert odds.bye["AE0"] == 1.0
    assert odds.mean_wins["AE0"] == len(afc)
