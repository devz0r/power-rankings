"""Quarterback ratings that follow the player, not the team.

A QB's rating is his EPA per play (dropbacks plus designed runs) above what
his opponents' pass defenses usually allow, across his whole career with
older plays counting less. It is shrunk toward what an unproven QB typically
does, so a backup with 30 career dropbacks isn't judged on one good game.

The team model's passing estimate already reflects whichever QBs played. So
the adjustment for a game is the difference between this game's starter and
the play-weighted mix of QBs behind the team's estimate.
"""

from __future__ import annotations

from collections import defaultdict

import polars as pl

from powerrank.data import Game


class QbTracker:
    """Defaults were chosen by cross-validation on 2006-2015; the error
    surface is flat around them."""

    def __init__(
        self,
        df: pl.DataFrame,
        half_life_weeks: float = 40.0,
        prior_plays: float = 500.0,
        prior_mean: float = -0.2,
    ):
        self.by_game: dict[str, list[dict]] = defaultdict(list)
        for row in df.iter_rows(named=True):
            self.by_game[row["game_id"]].append(row)
        self.half_life_weeks = half_life_weeks
        self.prior_plays = prior_plays
        # Relative to an average QB facing an average defense.
        self.prior_mean = prior_mean
        self._week = 0
        # qb -> [decayed value sum, decayed plays, week of last update]
        self._sums: dict[str, list[float]] = {}
        # Decayed league-wide EPA per QB play, so ratings are relative to the era.
        self._league = [0.0, 0.0, 0]

    def _decayed(self, entry: list[float]) -> tuple[float, float]:
        factor = 0.5 ** ((self._week - entry[2]) / self.half_life_weeks)
        return entry[0] * factor, entry[1] * factor

    def league_mean(self) -> float:
        epa, plays = self._decayed(self._league)
        return epa / plays if plays else 0.0

    def rating(self, qb: str | None) -> float:
        """EPA per play above average, shrunk toward an unproven QB."""
        value, plays = self._decayed(self._sums[qb]) if qb in self._sums else (0.0, 0.0)
        return (value + self.prior_plays * self.prior_mean) / (plays + self.prior_plays)

    def plays(self, qb: str) -> float:
        return self._decayed(self._sums[qb])[1] if qb in self._sums else 0.0

    def update_week(
        self,
        games: list[Game],
        defense_allowed: dict[str, float],
        fumble_epa: float,
        neutralize_fumbles: bool = True,
    ) -> list[dict]:
        """Absorb one week. `defense_allowed` is each defense's EPA per dropback
        allowed relative to average, as estimated before this week."""
        self._week += 1
        mean = self.league_mean()
        rows = [r for g in games for r in self.by_game.get(g.game_id, [])]
        for r in rows:
            epa = r["epa"]
            if neutralize_fumbles:
                epa += r["fumbles"] * fumble_epa - r["fumble_epa"]
            value = epa - r["plays"] * (mean + defense_allowed.get(r["defense"], 0.0))
            entry = self._sums.get(r["qb"])
            if entry is None:
                entry = self._sums[r["qb"]] = [0.0, 0.0, self._week]
            decayed = self._decayed(entry)
            entry[:] = [decayed[0] + value, decayed[1] + r["plays"], self._week]
            league = self._decayed(self._league)
            self._league = [league[0] + epa, league[1] + r["plays"], self._week]
        return rows
