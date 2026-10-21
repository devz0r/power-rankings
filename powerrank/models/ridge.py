"""Opponent-adjusted point differential, solved for all teams at once.

Each game says: home_margin - home_field = rating[home] - rating[away].
Ratings are the least-squares fit to this season's games, pulled toward a
prior (a fraction of last season's final rating). The pull is worth
`prior_games` games of evidence, so it dominates in September and fades as the
season fills in. Blowouts are capped so a 45-point win counts like a 28-point one.
"""

from __future__ import annotations

import numpy as np

from powerrank.data import Game
from powerrank.models.base import RatingModel


class RidgeMarginModel(RatingModel):
    name = "ridge"

    def __init__(
        self,
        prior_games: float = 8.0,
        carryover: float = 0.6,
        margin_cap: float = 28.0,
        half_life: float | None = None,
    ):
        super().__init__()
        self.prior_games = prior_games
        self.carryover = carryover
        self.margin_cap = margin_cap
        # Weeks after which a game counts half as much; None means no decay.
        self.half_life = half_life
        self._ratings: dict[str, float] = {}
        self._prior: dict[str, float] = {}
        self._season_games: list[Game] = []
        self._game_weeks: list[int] = []
        self._week_index = 0

    def start_season(self, season: int, teams: set[str]) -> None:
        prior = {t: self.carryover * self._ratings.get(t, 0.0) for t in teams}
        mean = sum(prior.values()) / len(prior)
        self._prior = {t: p - mean for t, p in prior.items()}
        self._ratings = dict(self._prior)
        self._season_games = []
        self._game_weeks = []
        self._week_index = 0

    def update_week(self, games: list[Game]) -> None:
        for g in games:
            self.hfa.update(g)
        self._season_games.extend(games)
        self._game_weeks.extend([self._week_index] * len(games))
        self._week_index += 1
        self._fit()

    def ratings(self) -> dict[str, float]:
        return dict(self._ratings)

    def _fit(self) -> None:
        teams = sorted(self._prior)
        index = {t: i for i, t in enumerate(teams)}
        n, m = len(teams), len(self._season_games)
        x = np.zeros((m, n))
        y = np.zeros(m)
        for row, g in enumerate(self._season_games):
            x[row, index[g.home]] = 1.0
            x[row, index[g.away]] = -1.0
            margin = np.clip(g.home_margin, -self.margin_cap, self.margin_cap)
            y[row] = margin - self.home_field(g)
        if self.half_life:
            ages = self._week_index - 1 - np.array(self._game_weeks)
            w = 0.5 ** (ages / self.half_life)
            x, y = x * np.sqrt(w)[:, None], y * np.sqrt(w)

        # Each team's prior counts as `prior_games` games of evidence; a game's
        # weight in the normal equations is 1 per team involved.
        lam = self.prior_games
        prior = np.array([self._prior[t] for t in teams])
        solution = np.linalg.solve(x.T @ x + lam * np.eye(n), x.T @ y + lam * prior)
        solution -= solution.mean()
        self._ratings = dict(zip(teams, solution.tolist()))
