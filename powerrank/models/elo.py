"""Elo baseline, after FiveThirtyEight's NFL model, expressed in points.

538 used K=20 Elo points with 25 Elo points per point of spread, a
margin-of-victory multiplier that dampens blowouts and autocorrelation, and a
one-third regression toward the mean between seasons. Tuning on 2006-2015
prefers 40%.
"""

from __future__ import annotations

import math

from powerrank.data import Game
from powerrank.models.base import RatingModel, win_probability

ELO_PER_POINT = 25.0


class EloModel(RatingModel):
    name = "elo"

    def __init__(self, k: float = 20.0, carryover: float = 0.6):
        super().__init__()
        self.k = k / ELO_PER_POINT
        self.carryover = carryover
        self._ratings: dict[str, float] = {}

    def start_season(self, season: int, teams: set[str]) -> None:
        self._ratings = {t: self.carryover * self._ratings.get(t, 0.0) for t in teams}
        self._center()

    def update_week(self, games: list[Game]) -> None:
        for g in games:
            expected = self.predict(g)
            margin = g.home_margin
            actual = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
            # Expected margin from the winner's side, in Elo points.
            winner_edge = (expected if margin >= 0 else -expected) * ELO_PER_POINT
            mov_mult = math.log(abs(margin) + 1) * 2.2 / (winner_edge * 0.001 + 2.2)
            shift = self.k * mov_mult * (actual - win_probability(expected))
            self._ratings[g.home] = self._ratings.get(g.home, 0.0) + shift
            self._ratings[g.away] = self._ratings.get(g.away, 0.0) - shift
            self.hfa.update(g)

    def ratings(self) -> dict[str, float]:
        return dict(self._ratings)

    def _center(self) -> None:
        if self._ratings:
            mean = sum(self._ratings.values()) / len(self._ratings)
            self._ratings = {t: r - mean for t, r in self._ratings.items()}
