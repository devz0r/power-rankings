"""Who is actually available at quarterback for upcoming games.

The schedule's projected starter can lag behind the news: a QB placed on
injured reserve can stay listed for the next game. nflverse's weekly rosters
carry each player's status, so a projected starter who isn't active that week
is replaced by the active QB with the most starts for the team this season.
Past games keep their real starters.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import replace

import polars as pl

from powerrank.data import CACHE_DIR, CACHE_MAX_AGE_HOURS, TEAM_ALIASES, Game

ACTIVE = "ACT"


def load_rosters(season: int, refresh: bool = False) -> pl.DataFrame:
    """QBs on each weekly roster: season, week, team, gsis_id, full_name, status."""
    path = CACHE_DIR / f"rosters_{season}.parquet"
    fresh = path.exists() and time.time() - path.stat().st_mtime < CACHE_MAX_AGE_HOURS * 3600
    if fresh and not refresh:
        return pl.read_parquet(path)
    import nflreadpy

    df = (
        nflreadpy.load_rosters_weekly([season])
        .filter(pl.col("position") == "QB")
        .select("season", "week", "team", "gsis_id", "full_name", "status")
        .with_columns(pl.col("team").replace(TEAM_ALIASES), pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32))
    )
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return df


class QbAvailability:
    def __init__(self, rosters: pl.DataFrame):
        self.status: dict[tuple[int, int, str, str], str] = {}
        self.names: dict[str, str] = {}
        self.latest_week: dict[tuple[int, str], int] = {}
        for season, week, team, gsis, name, status in rosters.iter_rows():
            if gsis is None:
                continue
            self.status[(season, week, team, gsis)] = status
            self.names[gsis] = name
            key = (season, team)
            self.latest_week[key] = max(self.latest_week.get(key, 0), week)

    def _week(self, season: int, week: int, team: str) -> int | None:
        """The roster week to use: `week`, or the latest published before it."""
        latest = self.latest_week.get((season, team))
        return None if latest is None else min(week, latest)

    def is_out(self, season: int, week: int, team: str, qb: str | None) -> bool:
        """True only when the roster says this QB isn't active for the team."""
        w = self._week(season, week, team)
        if w is None or qb is None:
            return False
        return self.status.get((season, w, team, qb)) != ACTIVE

    def active(self, season: int, week: int, team: str) -> list[str]:
        w = self._week(season, week, team)
        if w is None:
            return []
        return [q for (s, wk, t, q), st in self.status.items() if (s, wk, t) == (season, w, team) and st == ACTIVE]

    def replacement(self, season: int, week: int, team: str, played: list[Game]) -> str | None:
        """The active QB with the most starts for the team this season (most recent breaks ties)."""
        active = set(self.active(season, week, team))
        starts: Counter[str] = Counter()
        last_start: dict[str, int] = {}
        for g in played:
            for t, qb in ((g.home, g.home_qb), (g.away, g.away_qb)):
                if t == team and qb in active:
                    starts[qb] += 1
                    last_start[qb] = g.week
        if starts:
            return max(starts, key=lambda q: (starts[q], last_start[q]))
        return sorted(active)[0] if active else None


def correct_starters(games: list[Game], available: QbAvailability) -> list[Game]:
    """Replace projected starters who aren't active in unplayed games."""
    played_by_season: dict[int, list[Game]] = {}
    for g in games:
        if g.played:
            played_by_season.setdefault(g.season, []).append(g)
    fixed = []
    for g in games:
        if g.played:
            fixed.append(g)
            continue
        changes = {}
        for side, team, qb in (("home", g.home, g.home_qb), ("away", g.away, g.away_qb)):
            if qb is None or available.is_out(g.season, g.week, team, qb):
                sub = available.replacement(g.season, g.week, team, played_by_season.get(g.season, []))
                if sub and sub != qb:
                    changes[f"{side}_qb"] = sub
                    changes[f"{side}_qb_name"] = available.names.get(sub)
        fixed.append(replace(g, **changes) if changes else g)
    return fixed
