"""Walk-forward backtest: replay history one week at a time.

Before each week, the model predicts every game using only results from
earlier weeks. Then it sees that week's results. Predictions are scored
against actual margins and against the closing betting line, which is the
benchmark the model is trying to approach.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from powerrank.data import Game
from powerrank.models.base import RatingModel, win_probability


@dataclass(frozen=True)
class Prediction:
    game: Game
    predicted_margin: float

    @property
    def actual_margin(self) -> int:
        return self.game.home_margin


def group_weeks(games: Iterable[Game]) -> list[tuple[int, int, list[Game]]]:
    weeks: list[tuple[int, int, list[Game]]] = []
    for (season, week), group in itertools.groupby(games, key=lambda g: (g.season, g.week)):
        weeks.append((season, week, list(group)))
    return weeks


def run(
    model: RatingModel,
    games: list[Game],
    through_season: int | None = None,
    observe: Callable[[Game], None] | None = None,
) -> list[Prediction]:
    """Replay every played game in order, returning a pre-game prediction for each.

    `observe`, if given, is called for each game just before the model sees
    its result, so callers can capture any pre-game model state.
    """
    played = [g for g in games if g.played and (through_season is None or g.season <= through_season)]
    predictions: list[Prediction] = []
    current_season = None
    for season, _week, week_games in group_weeks(played):
        if season != current_season:
            teams = {t for g in played if g.season == season for t in (g.home, g.away)}
            model.start_season(season, teams)
            current_season = season
        for g in week_games:
            predictions.append(Prediction(g, model.predict(g)))
            if observe:
                observe(g)
        model.update_week(week_games)
    return predictions


@dataclass(frozen=True)
class Scores:
    games: int
    mae: float
    rmse: float
    log_loss: float
    brier: float
    accuracy: float

    def row(self) -> str:
        return (
            f"{self.games:>6}  {self.mae:6.2f}  {self.rmse:6.2f}  "
            f"{self.log_loss:7.4f}  {self.brier:6.4f}  {self.accuracy:6.1%}"
        )


SCORES_HEADER = " games     MAE    RMSE  logloss   brier   pick%"


def score(pairs: list[tuple[float, int]]) -> Scores:
    """Score (predicted margin, actual margin) pairs."""
    n = len(pairs)
    abs_err = sq_err = log_loss = brier = 0.0
    correct = decided = 0
    for pred, actual in pairs:
        abs_err += abs(pred - actual)
        sq_err += (pred - actual) ** 2
        p = min(max(win_probability(pred), 1e-6), 1 - 1e-6)
        outcome = 1.0 if actual > 0 else 0.0 if actual < 0 else 0.5
        log_loss -= outcome * math.log(p) + (1 - outcome) * math.log(1 - p)
        brier += (p - outcome) ** 2
        if actual != 0 and pred != 0:
            decided += 1
            correct += (pred > 0) == (actual > 0)
    return Scores(
        games=n,
        mae=abs_err / n,
        rmse=math.sqrt(sq_err / n),
        log_loss=log_loss / n,
        brier=brier / n,
        accuracy=correct / decided if decided else float("nan"),
    )


def evaluate(predictions: list[Prediction], seasons: range) -> tuple[Scores, Scores]:
    """Model and closing-line scores on the same games."""
    chosen = [p for p in predictions if p.game.season in seasons and p.game.spread_line is not None]
    model_scores = score([(p.predicted_margin, p.actual_margin) for p in chosen])
    market_scores = score([(p.game.spread_line, p.actual_margin) for p in chosen])
    return model_scores, market_scores


def tune(
    factory: Callable[..., RatingModel],
    grid: dict[str, list[float]],
    games: list[Game],
    seasons: range,
) -> list[tuple[dict[str, float], Scores]]:
    """Score every parameter combination on `seasons`, best (lowest MAE) first."""
    results = []
    names = list(grid)
    for values in itertools.product(*(grid[n] for n in names)):
        params = dict(zip(names, values))
        predictions = run(factory(**params), games, through_season=seasons.stop - 1)
        results.append((params, evaluate(predictions, seasons)[0]))
    return sorted(results, key=lambda item: item[1].mae)
