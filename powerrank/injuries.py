"""How much of each team's usual lineup is ruled out before a game.

For every team and week, sums the recent snap share (average over the team's
last four games) of each non-QB player listed Out or Doubtful on that week's
injury report. A starter who plays 95% of snaps counts ~0.95; a player who
has already been out for a month counts ~0, because the team's results have
already absorbed his absence. Quarterbacks are handled by the QB model.

Separate weights by position group (line, skill, front seven, secondary)
were tested and overfit: there are only a few seasons of training data, so
one combined measure is used. Injury reports start in 2009 and snap counts in
2012, so the index exists from 2012 on.
"""

from __future__ import annotations

import time
from collections import defaultdict

import polars as pl

from powerrank.data import CACHE_DIR, CACHE_MAX_AGE_HOURS, TEAM_ALIASES

FIRST_SEASON = 2012
RECENT_GAMES = 4
UNAVAILABLE = ("Out", "Doubtful")

OFFENSE = ("T", "G", "C", "OT", "OG", "OL", "WR", "TE", "RB", "FB", "HB")
DEFENSE = ("DE", "DT", "NT", "DL", "LB", "OLB", "ILB", "MLB", "EDGE", "CB", "S", "FS", "SS", "DB")


def _cached(name: str, season: int | None, loader, refresh: bool) -> pl.DataFrame:
    path = CACHE_DIR / (f"{name}_{season}.parquet" if season else f"{name}.parquet")
    current = season is None or season >= time.localtime().tm_year - 1
    fresh = path.exists() and not (current and time.time() - path.stat().st_mtime > CACHE_MAX_AGE_HOURS * 3600)
    if fresh and not (refresh and current):
        return pl.read_parquet(path)
    df = loader()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return df


def load_raw(seasons: range, refresh: bool = False) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    import nflreadpy

    injuries = pl.concat(
        [
            _cached("injuries", s, lambda s=s: nflreadpy.load_injuries([s]).select(
                "season", "week", "team", "gsis_id", "position", "report_status"
            ).with_columns(pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32)), refresh)
            for s in seasons
        ]
    )
    snaps = pl.concat(
        [
            _cached("snaps", s, lambda s=s: nflreadpy.load_snap_counts([s]).select(
                "season", "week", "team", "pfr_player_id", "offense_pct", "defense_pct"
            ).with_columns(pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32)), refresh)
            for s in seasons
        ]
    )
    players = _cached("players", None, lambda: nflreadpy.load_players().select("gsis_id", "pfr_id").drop_nulls(), refresh)
    return injuries, snaps, players


def build_index(injuries: pl.DataFrame, snaps: pl.DataFrame, players: pl.DataFrame) -> dict[tuple[int, int, str], float]:
    """(season, week, team) -> total recent snap share of players ruled out."""
    gsis_to_pfr = dict(players.iter_rows())
    share: dict[tuple[str, int, int], dict[str, tuple[float, float]]] = defaultdict(dict)
    for season, week, team, pfr, off, dfn in snaps.iter_rows():
        share[(TEAM_ALIASES.get(team, team), season, week)][pfr] = (off or 0.0, dfn or 0.0)
    team_games: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for team, season, week in sorted(share):
        team_games[team].append((season, week))

    index: dict[tuple[int, int, str], float] = defaultdict(float)
    out = injuries.filter(pl.col("report_status").is_in(UNAVAILABLE))
    for season, week, team, gsis, position, _status in out.iter_rows():
        pfr = gsis_to_pfr.get(gsis)
        team = TEAM_ALIASES.get(team, team)
        if position not in OFFENSE + DEFENSE or pfr is None:
            continue
        recent = [g for g in team_games[team] if g < (season, week)][-RECENT_GAMES:]
        if not recent:
            continue
        side = 0 if position in OFFENSE else 1
        index[(season, week, team)] += sum(share[(team, *g)].get(pfr, (0.0, 0.0))[side] for g in recent) / len(recent)
    return dict(index)


def load_index(last_season: int, refresh: bool = False) -> dict[tuple[int, int, str], float]:
    return build_index(*load_raw(range(FIRST_SEASON, last_season + 1), refresh=refresh))
