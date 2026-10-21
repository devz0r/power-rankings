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
    "qb_dropback", "passer_id", "rusher_id",
]

TABLES = ("team_games", "qb_games")


def _cache_path(table: str, season: int) -> Path:
    return CACHE_DIR / f"{table}_{season}.parquet"


def _clean(plays: pl.DataFrame) -> pl.DataFrame:
    return plays.select(COLUMNS).filter(
        pl.col("epa").is_not_null() & pl.col("posteam").is_not_null() & (pl.col("two_point_attempt") != 1)
    )


def aggregate(plays: pl.DataFrame) -> pl.DataFrame:
    plays = _clean(plays)
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


def aggregate_qbs(plays: pl.DataFrame) -> pl.DataFrame:
    """One row per (game, team, QB): his dropbacks plus his designed runs.

    A "QB" here is anyone who dropped back for the team in that game, so a
    receiver's trick-play pass counts too; such players have too few plays to
    matter once ratings are shrunk.
    """
    plays = _clean(plays).with_columns(
        pl.col("posteam").replace(TEAM_ALIASES).alias("team"),
        pl.col("defteam").replace(TEAM_ALIASES).alias("defense"),
    )
    dropbacks = plays.filter((pl.col("qb_dropback") == 1) & pl.col("passer_id").is_not_null()).with_columns(
        pl.col("passer_id").alias("qb")
    )
    qbs = dropbacks.select("game_id", "team", "qb").unique()
    runs = (
        plays.filter((pl.col("rush") == 1) & (pl.col("qb_dropback") != 1))
        .join(qbs, left_on=["game_id", "team", "rusher_id"], right_on=["game_id", "team", "qb"])
        .with_columns(pl.col("rusher_id").alias("qb"))
    )
    fumble = pl.col("fumble") == 1
    return (
        pl.concat([dropbacks.select(runs.columns), runs])
        .group_by(["game_id", "season", "week", "team", "defense", "qb"])
        .agg(
            pl.len().alias("plays"),
            pl.col("epa").sum().alias("epa"),
            fumble.sum().alias("fumbles"),
            pl.col("epa").filter(fumble).sum().alias("fumble_epa"),
        )
        .sort(["game_id", "team", "qb"])
    )


def load_season(season: int, refresh: bool = False, table: str = "team_games") -> pl.DataFrame:
    paths = {t: _cache_path(t, season) for t in TABLES}
    stale = any(
        not path.exists()
        or (
            season >= time.localtime().tm_year - 1
            and time.time() - path.stat().st_mtime > CACHE_MAX_AGE_HOURS * 3600
        )
        for path in paths.values()
    )
    if not stale and not refresh:
        return pl.read_parquet(paths[table])

    import nflreadpy

    plays = nflreadpy.load_pbp([season])
    tables = {"team_games": aggregate(plays), "qb_games": aggregate_qbs(plays)}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for t, df in tables.items():
        df.write_parquet(paths[t])
    return tables[table]


def load_team_games(seasons: range, refresh: bool = False, table: str = "team_games") -> pl.DataFrame:
    """`refresh` re-downloads only the latest season; finished seasons don't change."""
    latest = seasons.stop - 1
    return pl.concat([load_season(s, refresh=refresh and s == latest, table=table) for s in seasons])


def load_qb_games(seasons: range, refresh: bool = False) -> pl.DataFrame:
    return load_team_games(seasons, refresh=refresh, table="qb_games")
