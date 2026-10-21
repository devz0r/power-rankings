"""Play-by-play, aggregated to one row per (game, offense, play kind, garbage time).

Raw nflverse play-by-play is ~50k rows x 370 columns per season. The models
only need sums, so each season is reduced once and cached. Luck adjustments
(fumble recoveries, garbage-time weight) are applied later by the model, so
the cached sums stay raw and the adjustments stay tunable.

Kinds:
  pass    dropbacks (incl. scrambles and sacks), from the offense's view
  rush    designed runs
  kick    field goals and extra points, offense = kicking team
  return  kickoffs and punts, offense = receiving team
"""

from __future__ import annotations

import time
from pathlib import Path

import polars as pl

from powerrank.data import CACHE_DIR, CACHE_MAX_AGE_HOURS, TEAM_ALIASES

FIRST_SEASON = 1999

# Win probability outside this band counts as garbage time.
GARBAGE_WP = (0.1, 0.9)

COLUMNS = [
    "game_id", "season", "week", "posteam", "defteam", "home_team", "play_type",
    "epa", "wp", "pass", "rush", "fumble", "interception", "two_point_attempt",
]


def _cache_path(season: int) -> Path:
    return CACHE_DIR / f"team_games_{season}.parquet"


def aggregate(plays: pl.DataFrame) -> pl.DataFrame:
    plays = plays.select(COLUMNS).filter(
        pl.col("epa").is_not_null() & pl.col("posteam").is_not_null() & (pl.col("two_point_attempt") != 1)
    )
    kind = (
        pl.when(pl.col("pass") == 1).then(pl.lit("pass"))
        .when(pl.col("rush") == 1).then(pl.lit("rush"))
        .when(pl.col("play_type").is_in(["field_goal", "extra_point"])).then(pl.lit("kick"))
        .when(pl.col("play_type").is_in(["kickoff", "punt"])).then(pl.lit("return"))
    )
    low, high = GARBAGE_WP
    fumble = pl.col("fumble") == 1
    interception = pl.col("interception") == 1
    return (
        plays.with_columns(
            kind.alias("kind"),
            ((pl.col("wp") < low) | (pl.col("wp") > high)).fill_null(False).alias("garbage"),
            pl.col("posteam").replace(TEAM_ALIASES).alias("offense"),
            pl.col("defteam").replace(TEAM_ALIASES).alias("defense"),
            (pl.col("posteam") == pl.col("home_team")).alias("offense_home"),
        )
        .filter(pl.col("kind").is_not_null())
        .group_by(["game_id", "season", "week", "offense", "defense", "offense_home", "kind", "garbage"])
        .agg(
            pl.len().alias("plays"),
            pl.col("epa").sum().alias("epa"),
            fumble.sum().alias("fumbles"),
            pl.col("epa").filter(fumble).sum().alias("fumble_epa"),
            interception.sum().alias("interceptions"),
            pl.col("epa").filter(interception).sum().alias("interception_epa"),
        )
        .sort(["game_id", "offense", "kind", "garbage"])
    )


def load_season(season: int, refresh: bool = False) -> pl.DataFrame:
    path = _cache_path(season)
    stale = not path.exists() or (
        season >= time.localtime().tm_year - 1
        and time.time() - path.stat().st_mtime > CACHE_MAX_AGE_HOURS * 3600
    )
    if path.exists() and not stale and not refresh:
        return pl.read_parquet(path)

    import nflreadpy

    df = aggregate(nflreadpy.load_pbp([season]))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return df


def load_team_games(seasons: range, refresh: bool = False) -> pl.DataFrame:
    """`refresh` re-downloads only the latest season; finished seasons don't change."""
    latest = seasons.stop - 1
    return pl.concat([load_season(s, refresh=refresh and s == latest) for s in seasons])
