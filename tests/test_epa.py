import itertools
from datetime import date

import polars as pl
import pytest

from powerrank import backtest
from powerrank.__main__ import completed_through
from powerrank.data import Game
from powerrank.models.epa import EpaModel, TeamGameStats

OFFENSE = {"AAA": 0.20, "BBB": 0.05, "CCC": -0.05, "DDD": -0.20}
DEFENSE = {"AAA": -0.10, "BBB": 0.00, "CCC": 0.05, "DDD": 0.05}  # EPA allowed


def game(i: int, home: str, away: str, week: int, played: bool = True, gameday: str | None = None) -> Game:
    return Game(
        game_id=f"2020_{i}",
        season=2020,
        week=week,
        game_type="REG",
        home=home,
        away=away,
        neutral=False,
        gameday=gameday,
        home_score=20 if played else None,
        away_score=17 if played else None,
        spread_line=0.0,
    )


def row(g: Game, offense: str, defense: str, kind: str, epa: float, plays: int = 40, **extra) -> dict:
    base = {
        "game_id": g.game_id, "season": g.season, "week": g.week, "offense": offense, "defense": defense,
        "offense_home": offense == g.home, "kind": kind, "garbage": False, "plays": plays, "epa": epa,
        "fumbles": 0, "fumble_epa": 0.0, "interceptions": 0, "interception_epa": 0.0,
    }
    return base | extra


def round_robin() -> tuple[list[Game], list[dict]]:
    games, rows = [], []
    for i, (home, away) in enumerate(itertools.permutations(sorted(OFFENSE), 2)):
        g = game(i, home, away, week=i // 2 + 1)
        games.append(g)
        for off, dfn in ((home, away), (away, home)):
            per_play = OFFENSE[off] + DEFENSE[dfn]
            rows.append(row(g, off, dfn, "pass", per_play * 40))
            rows.append(row(g, off, dfn, "rush", per_play * 25, plays=25))
    return games, rows


def model_for(rows: list[dict], **kwargs) -> EpaModel:
    return EpaModel(stats=TeamGameStats(pl.DataFrame(rows)), weights={}, **kwargs)


def test_recovers_offense_and_defense_from_noiseless_games():
    games, rows = round_robin()
    model = model_for(rows, prior_plays=0.01, carryover=0.0)
    backtest.run(model, games)
    features = model.features()
    for team in OFFENSE:
        expected_off = OFFENSE[team] - sum(OFFENSE.values()) / 4
        expected_def = DEFENSE[team] - sum(DEFENSE.values()) / 4
        assert features[team]["off_pass"] == pytest.approx(expected_off, abs=1e-3)
        assert features[team]["def_pass"] == pytest.approx(expected_def, abs=1e-3)


def test_fumble_recovery_luck_is_removed():
    games, rows = round_robin()
    lost = [dict(r) for r in rows]
    recovered = [dict(r) for r in rows]
    # Same fumble, different bounce: lost costs -5 EPA, recovered costs -1.
    for version, cost in ((lost, -5.0), (recovered, -1.0)):
        version[0].update(fumbles=1, fumble_epa=cost, epa=version[0]["epa"] + cost)
        # Last season sets the league-average cost of a fumble (-3 EPA).
        version.append(row(game(99, "AAA", "BBB", week=1), "AAA", "BBB", "pass", -30.0) | {
            "game_id": "2019_99", "season": 2019, "fumbles": 10, "fumble_epa": -30.0,
        })

    results = []
    for version in (lost, recovered):
        model = model_for(version, prior_plays=10)
        backtest.run(model, games)
        results.append(model.features()["AAA"]["off_pass"])
    assert results[0] == pytest.approx(results[1])

    unadjusted = []
    for version in (lost, recovered):
        model = model_for(version, prior_plays=10, neutralize_fumbles=False)
        backtest.run(model, games)
        unadjusted.append(model.features()["AAA"]["off_pass"])
    assert unadjusted[0] < unadjusted[1]


def test_breakdown_sums_to_rating():
    games, rows = round_robin()
    model = model_for(rows)
    model.weights = {"off_pass": 10.0, "def_pass": -8.0, "off_rush": 5.0, "def_rush": -4.0, "margin": 0.5}
    backtest.run(model, games)
    for team, rating in model.ratings().items():
        assert sum(model.breakdown(team).values()) == pytest.approx(rating)


def test_completed_through_waits_for_the_whole_week():
    today = date(2026, 10, 20)
    games = [
        game(1, "AAA", "BBB", week=1, gameday="2026-10-04"),
        game(2, "CCC", "DDD", week=2, gameday="2026-10-11"),
        game(3, "AAA", "CCC", week=3, gameday="2026-10-18"),
        game(4, "BBB", "DDD", week=3, played=False, gameday="2026-10-19"),  # Monday night, not scored yet
    ]
    assert completed_through(games, today) == 2
    assert completed_through(games[:3], today) == 3


def test_completed_through_skips_long_unplayed_games():
    games = [
        game(1, "AAA", "BBB", week=1, played=False, gameday="2026-09-10"),  # cancelled
        game(2, "CCC", "DDD", week=2, gameday="2026-09-17"),
    ]
    assert completed_through(games, date(2026, 10, 1)) == 2
