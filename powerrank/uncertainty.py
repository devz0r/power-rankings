"""How far a team's rating may be from how it actually plays.

Measured, not assumed: replay history and compare each team's rating at week
w with its average opponent-adjusted margin over the rest of that season.
That average carries game-to-game noise of sigma^2 / n (n games remaining)
on top of the rating's own error, so regressing squared differences on 1/n
separates the two. What's left includes in-season change (injuries, form),
which is exactly the uncertainty a reader cares about.

On 2006-2025 the game noise is ~13.5 points and the rating uncertainty is
~4.3 points before the season, settling near ~3.7 from midseason on: teams
keep changing, so it never shrinks to zero.
"""

from __future__ import annotations

import numpy as np

from powerrank import backtest
from powerrank.data import Game
from powerrank.models.base import RatingModel

# Weeks grouped so each bucket has enough team-weeks to estimate from.
BUCKETS = ((1, 1), (2, 3), (4, 6), (7, 9), (10, 12), (13, 99))


def calibrate(model: RatingModel, games: list[Game], seasons: range) -> dict:
    """Replay `games` with `model` and return game noise and rating sd by week."""
    played = [g for g in games if g.played and g.game_type == "REG" and g.season < seasons.stop]
    snapshots: dict[tuple[int, int], tuple[dict[str, float], float]] = {}
    current = None
    for season, week, week_games in backtest.group_weeks(played):
        if season != current:
            model.start_season(season, {t for g in played if g.season == season for t in (g.home, g.away)})
            current = season
        snapshots[(season, week)] = (model.ratings(), model.hfa.value)
        model.update_week(week_games)

    by_season: dict[int, list[Game]] = {}
    for g in played:
        by_season.setdefault(g.season, []).append(g)
    rows = []
    for (season, week), (ratings, hfa) in snapshots.items():
        if season not in seasons:
            continue
        future = [g for g in by_season[season] if g.week >= week]
        for team, rating in ratings.items():
            perf = [
                (g.home_margin - (0 if g.neutral else hfa)) * (1 if g.home == team else -1)
                + ratings[g.away if g.home == team else g.home]
                for g in future
                if team in (g.home, g.away)
            ]
            if len(perf) >= 2:
                rows.append((week, len(perf), (np.mean(perf) - rating) ** 2))

    data = np.array(rows)
    x = np.column_stack([np.ones(len(data)), 1 / data[:, 1]])
    (_, game_var), *_ = np.linalg.lstsq(x, data[:, 2], rcond=None)
    sd_by_bucket = []
    for lo, hi in BUCKETS:
        m = (data[:, 0] >= lo) & (data[:, 0] <= hi)
        rating_var = float(np.mean(data[m, 2] - game_var / data[m, 1]))
        sd_by_bucket.append([lo, hi, round(float(np.sqrt(max(rating_var, 0.0))), 2)])
    return {"game_sd": round(float(np.sqrt(game_var)), 2), "rating_sd": sd_by_bucket}


def rating_sd(calibration: dict, week: int) -> float:
    """Rating uncertainty for ratings made before `week`."""
    for lo, hi, sd in calibration["rating_sd"]:
        if lo <= week <= hi:
            return sd
    return calibration["rating_sd"][-1][2]
