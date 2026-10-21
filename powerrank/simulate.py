"""Monte Carlo of the rest of the season: playoff, division, bye and title odds.

Each simulation draws every team's true strength once from its rating's
uncertainty (so a team that is secretly better stays better all season), then
plays each remaining game with the usual game-to-game noise. Standings use
simplified NFL tiebreakers: win percentage, then head-to-head among the tied
teams, then division record (for division races) or conference record, then
a coin flip. The real rules go on to common games and strength of victory,
which rarely decide anything the simplified order doesn't.

Playoffs: 7 seeds per conference, 4 division winners first, only the 1 seed
gets a bye, the higher seed hosts, the Super Bowl is neutral. Playoff games
already played use their real results.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from powerrank.data import Game
from powerrank.teams import Team

N_SIMS = 10_000


@dataclass
class SeasonOdds:
    mean_wins: dict[str, float]
    playoffs: dict[str, float]
    division: dict[str, float]
    bye: dict[str, float]
    conference: dict[str, float]
    super_bowl: dict[str, float]


def simulate(
    teams: dict[str, Team],
    played: list[Game],
    remaining: list[tuple[Game, float]],
    strength: dict[str, float],
    rating_sd: float,
    game_sd: float,
    hfa: float,
    n_sims: int = N_SIMS,
    seed: int = 0,
) -> SeasonOdds:
    """`remaining` pairs each unplayed regular-season game with the expected
    home margin; `strength` is each team's rating for playoff games."""
    rng = np.random.default_rng(seed)
    abbrs = sorted(teams)
    idx = {t: i for i, t in enumerate(abbrs)}
    n = len(abbrs)
    conf = np.array([teams[t].conference for t in abbrs])
    div = np.array([teams[t].division for t in abbrs])
    delta = rng.normal(0.0, rating_sd, (n_sims, n))

    # Results: wins (ties count half) overall, in division and in conference,
    # and head-to-head wins between every pair.
    wins = np.zeros((n_sims, n))
    games_played = np.zeros(n)
    div_wins, div_games = np.zeros((n_sims, n)), np.zeros(n)
    conf_wins, conf_games = np.zeros((n_sims, n)), np.zeros(n)
    h2h = np.zeros((n_sims, n, n))

    def add(h: int, a: int, home_result: np.ndarray | float) -> None:
        away_result = 1.0 - home_result
        wins[:, h] += home_result
        wins[:, a] += away_result
        games_played[[h, a]] += 1
        h2h[:, h, a] += home_result
        h2h[:, a, h] += away_result
        if div[h] == div[a]:
            div_wins[:, h] += home_result
            div_wins[:, a] += away_result
            div_games[[h, a]] += 1
        if conf[h] == conf[a]:
            conf_wins[:, h] += home_result
            conf_wins[:, a] += away_result
            conf_games[[h, a]] += 1

    for g in played:
        if g.game_type == "REG":
            add(idx[g.home], idx[g.away], 1.0 if g.home_margin > 0 else 0.0 if g.home_margin < 0 else 0.5)
    for g, mean in remaining:
        h, a = idx[g.home], idx[g.away]
        margin = mean + delta[:, h] - delta[:, a] + rng.normal(0.0, game_sd, n_sims)
        add(h, a, (margin > 0).astype(float))

    pct = wins / np.maximum(games_played, 1)
    div_pct = div_wins / np.maximum(div_games, 1)
    conf_pct = conf_wins / np.maximum(conf_games, 1)
    coin = rng.random((n_sims, n))
    base = np.array([strength[t] for t in abbrs])
    actual = {frozenset((g.home, g.away)): (g.home if g.home_margin > 0 else g.away)
              for g in played if g.game_type != "REG"}

    counts = {k: np.zeros(n) for k in ("playoffs", "division", "bye", "conference", "super_bowl")}
    for s in range(n_sims):
        champions = []
        for c in sorted(set(conf)):
            seeds = _seed_conference(
                [i for i in range(n) if conf[i] == c], div, pct[s], div_pct[s], conf_pct[s], h2h[s], coin[s]
            )
            for rank, i in enumerate(seeds):
                counts["playoffs"][i] += 1
                if rank < 4:
                    counts["division"][i] += 1
            counts["bye"][seeds[0]] += 1
            winner = _play_conference(seeds, base + delta[s], hfa, game_sd, rng, actual, abbrs)
            counts["conference"][winner] += 1
            champions.append(winner)
        sb = _play(champions[0], champions[1], base + delta[s], 0.0, game_sd, rng, actual, abbrs)
        counts["super_bowl"][sb] += 1

    as_dict = lambda arr: {t: float(arr[idx[t]]) for t in abbrs}
    return SeasonOdds(
        mean_wins=as_dict(wins.mean(axis=0)),
        **{k: as_dict(v / n_sims) for k, v in counts.items()},
    )


def _rank(members: list[int], pct, secondary, h2h, coin) -> list[int]:
    """Order teams: win pct, head-to-head among teams tied on win pct,
    then `secondary` (division or conference record), then a coin flip."""
    groups: dict[float, list[int]] = defaultdict(list)
    for i in members:
        groups[round(float(pct[i]), 6)].append(i)
    h2h_pct = {}
    for group in groups.values():
        for i in group:
            others = [j for j in group if j != i]
            played = sum(h2h[i, j] + h2h[j, i] for j in others)
            h2h_pct[i] = sum(h2h[i, j] for j in others) / played if played else 0.5
    return sorted(members, key=lambda i: (pct[i], h2h_pct[i], secondary[i], coin[i]), reverse=True)


def _seed_conference(members, div, pct, div_pct, conf_pct, h2h, coin) -> list[int]:
    winners = [
        _rank([i for i in members if div[i] == d], pct, div_pct, h2h, coin)[0]
        for d in sorted({div[i] for i in members})
    ]
    others = [i for i in members if i not in winners]
    return _rank(winners, pct, conf_pct, h2h, coin) + _rank(others, pct, conf_pct, h2h, coin)[:3]


def _play(home: int, away: int, strength, hfa, game_sd, rng, actual, abbrs) -> int:
    result = actual.get(frozenset((abbrs[home], abbrs[away])))
    if result is not None:
        return home if result == abbrs[home] else away
    margin = strength[home] - strength[away] + hfa + rng.normal(0.0, game_sd)
    return home if margin > 0 else away


def _play_conference(seeds: list[int], strength, hfa, game_sd, rng, actual, abbrs) -> int:
    seed_of = {team: rank for rank, team in enumerate(seeds)}
    alive = [seeds[0]] + [
        _play(seeds[hi], seeds[lo], strength, hfa, game_sd, rng, actual, abbrs) for hi, lo in ((1, 6), (2, 5), (3, 4))
    ]
    while len(alive) > 1:
        alive.sort(key=seed_of.get)
        # Best remaining seed hosts the worst; the middle two meet.
        pairs = [(alive[0], alive[-1])] + ([(alive[1], alive[2])] if len(alive) == 4 else [])
        alive = [_play(h, a, strength, hfa, game_sd, rng, actual, abbrs) for h, a in pairs]
    return alive[0]
