import itertools
import random
from dataclasses import replace

import polars as pl
import pytest

from powerrank import backtest
from powerrank.data import Game, to_games
from powerrank.models import EloModel, RidgeMarginModel
from powerrank.models.base import win_probability

TRUE_RATINGS = {"AAA": 7.0, "BBB": 3.0, "CCC": -1.0, "DDD": -9.0}


def make_season(season: int, ratings: dict[str, float], hfa: float = 0.0, noise: float = 0.0, seed: int = 0):
    """Double round robin, one game per team per week."""
    rng = random.Random(seed)
    games = []
    for i, (home, away) in enumerate(itertools.permutations(sorted(ratings), 2)):
        margin = round(ratings[home] - ratings[away] + hfa + rng.gauss(0, noise))
        games.append(
            Game(
                game_id=f"{season}_{i}",
                season=season,
                week=i // 2 + 1,
                game_type="REG",
                home=home,
                away=away,
                neutral=False,
                gameday=None,
                home_score=30 + margin if margin > 0 else 30,
                away_score=30 if margin > 0 else 30 - margin,
                spread_line=0.0,
            )
        )
    return games


def test_win_probability_is_symmetric():
    assert win_probability(0) == pytest.approx(0.5)
    assert win_probability(7) == pytest.approx(1 - win_probability(-7))
    assert 0.5 < win_probability(3) < win_probability(10) < 1


def test_ridge_recovers_true_ratings_with_weak_prior():
    model = RidgeMarginModel(prior_games=0.01, carryover=0.0, margin_cap=100)
    backtest.run(model, make_season(2020, TRUE_RATINGS))
    for team, rating in TRUE_RATINGS.items():
        assert model.ratings()[team] == pytest.approx(rating - sum(TRUE_RATINGS.values()) / 4, abs=0.05)


def test_ridge_prior_pulls_early_ratings_toward_last_season():
    model = RidgeMarginModel(prior_games=8, carryover=0.5)
    model.start_season(2020, set(TRUE_RATINGS))
    model._ratings = {"AAA": 10.0, "BBB": 0.0, "CCC": 0.0, "DDD": -10.0}
    model.start_season(2021, set(TRUE_RATINGS))
    assert model.ratings()["AAA"] == pytest.approx(5.0)
    assert model.ratings()["DDD"] == pytest.approx(-5.0)


@pytest.mark.parametrize("model_cls", [EloModel, RidgeMarginModel])
def test_ratings_are_centered_and_order_teams(model_cls):
    model = model_cls()
    games = make_season(2020, TRUE_RATINGS, noise=3, seed=1) + make_season(2021, TRUE_RATINGS, noise=3, seed=2)
    backtest.run(model, games)
    ratings = model.ratings()
    assert sum(ratings.values()) == pytest.approx(0, abs=1e-9)
    assert sorted(ratings, key=ratings.get, reverse=True) == ["AAA", "BBB", "CCC", "DDD"]


@pytest.mark.parametrize("model_cls", [EloModel, RidgeMarginModel])
def test_predictions_never_see_their_own_result(model_cls):
    games = make_season(2020, TRUE_RATINGS, noise=10, seed=3)
    target = 5
    flipped = list(games)
    flipped[target] = replace(games[target], home_score=games[target].away_score, away_score=games[target].home_score)
    if games[target].home_margin == 0:
        flipped[target] = replace(games[target], home_score=games[target].home_score + 20)

    original = backtest.run(model_cls(), games)
    changed = backtest.run(model_cls(), flipped)
    same_week = [i for i, g in enumerate(games) if g.week <= games[target].week]
    later = [i for i, g in enumerate(games) if g.week > games[target].week]
    for i in same_week:
        assert original[i].predicted_margin == changed[i].predicted_margin
    assert any(original[i].predicted_margin != changed[i].predicted_margin for i in later)


def test_home_field_estimate_tracks_average_home_margin():
    model = RidgeMarginModel()
    games = make_season(2020, TRUE_RATINGS, hfa=3.0) * 10
    for i, g in enumerate(games):
        games[i] = replace(g, week=i // 2 + 1)
    backtest.run(model, games)
    assert model.hfa.value == pytest.approx(3.0, abs=0.5)


def test_unplayed_games_are_skipped():
    games = make_season(2020, TRUE_RATINGS)
    games[-1] = replace(games[-1], home_score=None, away_score=None)
    assert len(backtest.run(EloModel(), games)) == len(games) - 1


def test_relocated_teams_use_current_abbreviation():
    df = pl.DataFrame(
        {
            "game_id": ["g1"],
            "season": [2010],
            "week": [1],
            "gameday": ["2010-09-12"],
            "gametime": ["13:00"],
            "game_type": ["REG"],
            "home_team": ["OAK"],
            "away_team": ["SD"],
            "location": ["Home"],
            "home_score": [10],
            "away_score": [7],
            "spread_line": [-3.0],
        }
    )
    (game,) = to_games(df)
    assert (game.home, game.away) == ("LV", "LAC")


def test_scores_for_perfect_predictions():
    s = backtest.score([(7.0, 7), (-3.0, -3)])
    assert s.mae == 0 and s.rmse == 0 and s.accuracy == 1
