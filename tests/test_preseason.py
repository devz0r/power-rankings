import random

import pytest

from powerrank.data import Game
from powerrank.models.epa import FEATURES
from powerrank.preseason import Preseason, SeasonEstimates, offseason_info


def game(season: int, week: int, home: str, away: str, home_qb: str, away_qb: str, home_coach="H", away_coach="A"):
    return Game(
        game_id=f"{season}_{week}_{home}", season=season, week=week, game_type="REG", home=home, away=away,
        neutral=False, gameday=None, home_score=21, away_score=14, spread_line=0.0,
        home_qb=home_qb, away_qb=away_qb, home_coach=home_coach, away_coach=away_coach,
    )


def test_offseason_detects_qb_and_coach_changes():
    games = []
    # 2019-2020: AAA starts qb1 under coach X; BBB starts veteran vet1.
    for season in (2019, 2020):
        for week in range(1, 10):
            games.append(game(season, week, "AAA", "BBB", "qb1", "vet1", home_coach="X", away_coach="Y"))
    # 2021: AAA's starter is vet1 (18 career starts) and a new coach; BBB starts a rookie.
    games.append(game(2021, 1, "AAA", "BBB", "vet1", "rookie", home_coach="Z", away_coach="Y"))

    info = offseason_info(games, drafts={})
    assert info[(2020, "AAA")] == {
        "qb_changed": 0.0, "veteran_qb": 0.0, "qb_delta": 0.0, "coach_changed": 0.0, "draft": 0.0,
    }
    assert info[(2021, "AAA")]["qb_changed"] == 1.0
    assert info[(2021, "AAA")]["veteran_qb"] == 1.0
    assert info[(2021, "AAA")]["coach_changed"] == 1.0
    assert info[(2021, "BBB")]["qb_changed"] == 1.0
    assert info[(2021, "BBB")]["veteran_qb"] == 0.0
    assert info[(2021, "BBB")]["coach_changed"] == 0.0


def test_draft_capital_is_relative_to_the_league():
    games = [game(2020, 1, "AAA", "BBB", "q1", "q2")]
    info = offseason_info(games, drafts={(2020, "AAA"): 3.0, (2020, "BBB"): 1.0})
    assert info[(2020, "AAA")]["draft"] == pytest.approx(1.0)
    assert info[(2020, "BBB")]["draft"] == pytest.approx(-1.0)


def synthetic(carry: float, qb_penalty: float, seed: int = 0):
    """Each feature = carry * last season, minus `qb_penalty` of it when the QB changed, plus noise."""
    rng = random.Random(seed)
    teams = [f"T{i:02d}" for i in range(32)]
    values = {2000: {t: {f: rng.gauss(0, 1) for f in FEATURES} for t in teams}}
    values[1999] = {t: {f: 0.0 for f in FEATURES} for t in teams}
    offseason = {}
    for season in range(2001, 2031):
        values[season] = {}
        for t in teams:
            changed = float(rng.random() < 0.25)
            offseason[(season, t)] = {
                "qb_changed": changed, "veteran_qb": 0.0, "qb_delta": 0.0, "coach_changed": 0.0, "draft": 0.0,
            }
            values[season][t] = {
                f: (carry - qb_penalty * changed) * values[season - 1][t][f] + rng.gauss(0, 0.5) for f in FEATURES
            }
    noise = dict.fromkeys(FEATURES, 1.0)
    sampling = dict.fromkeys(FEATURES, 0.0)
    return SeasonEstimates(values, noise, sampling), offseason, teams


def test_projection_learns_carryover_and_qb_discount():
    estimates, offseason, _ = synthetic(carry=0.6, qb_penalty=0.4)
    pre = Preseason.fit(estimates, offseason, range(2002, 2031))
    last, last2, qb_x_last = pre.coefficients["off_pass"][:3]
    assert last == pytest.approx(0.6, abs=0.05)
    assert last2 == pytest.approx(0.0, abs=0.05)
    assert qb_x_last == pytest.approx(-0.4, abs=0.08)


def test_prior_strength_is_noise_over_projection_error():
    estimates, offseason, _ = synthetic(carry=0.6, qb_penalty=0.0)
    pre = Preseason.fit(estimates, offseason, range(2002, 2031))
    # Projection error has variance 0.25 and play noise is 1, so a prior is worth ~4 plays.
    assert pre.strengths["off_pass"] == pytest.approx(4.0, rel=0.15)


def test_projections_are_centered():
    estimates, offseason, teams = synthetic(carry=0.6, qb_penalty=0.2)
    pre = Preseason.fit(estimates, offseason, range(2002, 2030))
    projected = pre.project(2030, teams)
    for f in FEATURES:
        assert sum(projected[t][f] for t in teams) == pytest.approx(0.0, abs=1e-9)
