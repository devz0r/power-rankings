"""Common interface for rating models.

Every rating is in points: the margin a team would beat a league-average team
by on a neutral field. The backtest drives a model one week at a time:

    start_season -> predict (all games in week) -> update_week -> ...

so a model can never see a game's result before predicting it.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import deque

from powerrank.data import Game

# Standard deviation of NFL game margins around the expected margin, used to
# turn a predicted margin into a win probability.
MARGIN_SD = 13.5


def win_probability(margin: float, sd: float = MARGIN_SD) -> float:
    return 0.5 * (1.0 + math.erf(margin / (sd * math.sqrt(2.0))))


class HomeFieldEstimator:
    """Home-field advantage as the average home margin over recent non-neutral games.

    Home and away teams are balanced over a full schedule, so the average home
    margin is an unbiased estimate of the advantage, and a trailing window
    tracks its decline over time.
    """

    def __init__(self, window: int = 800, initial: float = 2.5):
        self.margins: deque[int] = deque(maxlen=window)
        self.initial = initial

    @property
    def value(self) -> float:
        if len(self.margins) < 100:
            return self.initial
        return sum(self.margins) / len(self.margins)

    def update(self, game: Game) -> None:
        if not game.neutral:
            self.margins.append(game.home_margin)


class RatingModel(ABC):
    name: str = "base"

    def __init__(self) -> None:
        self.hfa = HomeFieldEstimator()

    @abstractmethod
    def start_season(self, season: int, teams: set[str]) -> None: ...

    @abstractmethod
    def update_week(self, games: list[Game]) -> None:
        """Absorb the results of one completed week."""

    @abstractmethod
    def ratings(self) -> dict[str, float]: ...

    def home_field(self, game: Game) -> float:
        return 0.0 if game.neutral else self.hfa.value

    def predict(self, game: Game) -> float:
        """Expected home margin."""
        r = self.ratings()
        return r.get(game.home, 0.0) - r.get(game.away, 0.0) + self.home_field(game)
