"""Team names, conferences, divisions, colors and logos."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from powerrank.data import CACHE_DIR, TEAM_ALIASES

# Old abbreviations, plus a duplicate Rams row (schedules use "LA").
NOT_CURRENT = set(TEAM_ALIASES) | {"LAR"}
# nflverse calls the Rams "LA", which is ambiguous next to the Chargers.
DISPLAY_ABBR = {"LA": "LAR"}


@dataclass(frozen=True)
class Team:
    abbr: str
    name: str
    nickname: str
    conference: str
    division: str
    color: str
    color2: str
    logo: str

    @property
    def display_abbr(self) -> str:
        return DISPLAY_ABBR.get(self.abbr, self.abbr)


def load_teams(refresh: bool = False) -> dict[str, Team]:
    path = CACHE_DIR / "teams.parquet"
    if path.exists() and not refresh:
        df = pl.read_parquet(path)
    else:
        import nflreadpy

        df = nflreadpy.load_teams()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.write_parquet(path)
    return {
        r["team_abbr"]: Team(
            abbr=r["team_abbr"],
            name=r["team_name"],
            nickname=r["team_nick"],
            conference=r["team_conf"],
            division=r["team_division"],
            color=r["team_color"],
            color2=r["team_color2"],
            logo=r["team_logo_espn"],
        )
        for r in df.iter_rows(named=True)
        if r["team_abbr"] not in NOT_CURRENT
    }
