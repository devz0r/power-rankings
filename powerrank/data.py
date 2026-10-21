"""Loading and normalizing nflverse data."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import polars as pl

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
CACHE_MAX_AGE_HOURS = 6

# Relocated franchises are tracked under their current abbreviation so ratings
# carry across moves.
TEAM_ALIASES = {"OAK": "LV", "SD": "LAC", "STL": "LA"}


@dataclass(frozen=True)
class Game:
    game_id: str
    season: int
    week: int
    game_type: str
    home: str
    away: str
    neutral: bool
    gameday: str | None
    home_score: int | None
    away_score: int | None
    # nflverse convention: positive means the home team is favored by that many points.
    spread_line: float | None
    # Starting QBs (gsis ids) and head coaches, known before kickoff.
    home_qb: str | None = None
    away_qb: str | None = None
    home_coach: str | None = None
    away_coach: str | None = None
    home_qb_name: str | None = None
    away_qb_name: str | None = None

    @property
    def played(self) -> bool:
        return self.home_score is not None and self.away_score is not None

    @property
    def home_margin(self) -> int:
        if not self.played:
            raise ValueError(f"{self.game_id} has not been played")
        return self.home_score - self.away_score


def load_schedules(refresh: bool = False) -> pl.DataFrame:
    """All games since 1999, cached locally for a few hours."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / "schedules.parquet"
    fresh = path.exists() and (time.time() - path.stat().st_mtime) < CACHE_MAX_AGE_HOURS * 3600
    if fresh and not refresh:
        return pl.read_parquet(path)

    import nflreadpy

    df = nflreadpy.load_schedules()
    df.write_parquet(path)
    return df


def to_games(df: pl.DataFrame) -> list[Game]:
    """Convert a schedules frame to Game objects, sorted chronologically."""
    df = df.with_columns(
        pl.col("home_team").replace(TEAM_ALIASES),
        pl.col("away_team").replace(TEAM_ALIASES),
    ).sort(["season", "week", "gameday", "gametime", "game_id"])
    return [
        Game(
            game_id=r["game_id"],
            season=r["season"],
            week=r["week"],
            game_type=r["game_type"],
            home=r["home_team"],
            away=r["away_team"],
            neutral=r["location"] == "Neutral",
            gameday=r["gameday"],
            home_score=r["home_score"],
            away_score=r["away_score"],
            spread_line=r["spread_line"],
            home_qb=r.get("home_qb_id"),
            away_qb=r.get("away_qb_id"),
            home_coach=r.get("home_coach"),
            away_coach=r.get("away_coach"),
            home_qb_name=r.get("home_qb_name"),
            away_qb_name=r.get("away_qb_name"),
        )
        for r in df.iter_rows(named=True)
    ]


def load_games(refresh: bool = False) -> list[Game]:
    return to_games(load_schedules(refresh=refresh))
