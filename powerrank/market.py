"""What Vegas thinks each team is worth, for the disagreement column.

Point spreads say: spread = rating[home] - rating[away] + home field. Solving
that for every team across this season's lines gives market-implied ratings
on the same points scale as ours. Recent lines count more (half-life of two
weeks), because the market reprices teams as QBs get hurt or return, and the
coming week's lines carry the current starters. These are never fed to the
model; they are only compared against it.
"""

from __future__ import annotations

import numpy as np

from powerrank.data import Game

HALF_LIFE_WEEKS = 2.0
# A token pull toward average, only for numerical stability: lines already
# are the market's full view of each team, so shrinking them further would
# just compress the market's ratings (0.5 made them ~20% too narrow).
PRIOR_GAMES = 0.05


def implied_ratings(season_games: list[Game], latest_week: int) -> tuple[dict[str, float], float]:
    """Market ratings and home-field edge from lines through `latest_week`."""
    games = [g for g in season_games if g.spread_line is not None and g.week <= latest_week]
    teams = sorted({t for g in games for t in (g.home, g.away)})
    index = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    x = np.zeros((len(games), n + 1))
    y = np.empty(len(games))
    w = np.empty(len(games))
    for row, g in enumerate(games):
        x[row, index[g.home]] = 1.0
        x[row, index[g.away]] = -1.0
        x[row, n] = 0.0 if g.neutral else 1.0
        y[row] = g.spread_line
        w[row] = 0.5 ** ((latest_week - g.week) / HALF_LIFE_WEEKS)
    penalty = np.full(n + 1, PRIOR_GAMES)
    penalty[n] = 1e-6
    a = x.T @ (x * w[:, None]) + np.diag(penalty)
    solution = np.linalg.solve(a, x.T @ (w * y))
    ratings = solution[:n] - solution[:n].mean()
    return dict(zip(teams, ratings.tolist())), float(solution[n])
