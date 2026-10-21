"""What teams have accomplished: shown beside the ranking, never used for it.

Past games judge each opponent as it was fielded that day: its current
rating (the best estimate of how good the team is), adjusted for the QB who
started that game and the players ruled out that week. Beating a team with
its starter counts as beating it with its starter, even if he's hurt now.
The remaining schedule uses each team's expected healthy lineup.
"""

from __future__ import annotations

from powerrank.data import Game
from powerrank.models.base import win_probability


def record(team: str, games: list[Game]) -> dict:
    wins = losses = ties = points_for = points_against = 0
    for g in games:
        if team not in (g.home, g.away):
            continue
        margin = g.home_margin if g.home == team else -g.home_margin
        wins += margin > 0
        losses += margin < 0
        ties += margin == 0
        scored = g.home_score if g.home == team else g.away_score
        points_for += scored
        points_against += scored - margin
    return {"wins": wins, "losses": losses, "ties": ties, "point_diff": points_for - points_against}


def _edge(team: str, g: Game, hfa: float) -> float:
    """Home-field points from `team`'s side of the game."""
    if g.neutral:
        return 0.0
    return hfa if g.home == team else -hfa


def _opponent(team: str, g: Game) -> str:
    return g.away if g.home == team else g.home


def strength_of_record(
    team: str, played: list[Game], faced: dict[tuple[str, str], float], hfa: float, reference: float
) -> float:
    """Chance a team rated `reference` would match or beat this record
    against this schedule. Lower means more impressive. `faced` maps
    (game_id, team) to that team's strength in that game."""
    games = [g for g in played if team in (g.home, g.away)]
    won = sum(1.0 if (g.home_margin if g.home == team else -g.home_margin) > 0 else
              0.5 if g.home_margin == 0 else 0.0 for g in games)
    # Distribution of the reference team's wins (Poisson-binomial).
    dist = [1.0]
    for g in games:
        p = win_probability(reference - faced[(g.game_id, _opponent(team, g))] + _edge(team, g, hfa))
        nxt = [0.0] * (len(dist) + 1)
        for k, prob in enumerate(dist):
            nxt[k] += prob * (1 - p)
            nxt[k + 1] += prob * p
        dist = nxt
    return sum(prob for k, prob in enumerate(dist) if k >= won - 1e-9)


def notable_results(
    team: str, played: list[Game], faced: dict[tuple[str, str], float], teams: dict | None = None
) -> dict:
    """Best win (strongest opponent beaten) and worst loss (weakest opponent
    lost to), judging each opponent as it was fielded in that game."""
    wins, losses = [], []
    for g in played:
        if team not in (g.home, g.away):
            continue
        margin = g.home_margin if g.home == team else -g.home_margin
        opp = _opponent(team, g)
        info = (teams or {}).get(opp)
        entry = {
            "opponent": opp,
            "opponent_abbr": info.display_abbr if info else opp,
            "opponent_name": info.nickname if info else opp,
            "opponent_starter": g.away_qb_name if g.home == team else g.home_qb_name,
            "score": f"{max(g.home_score, g.away_score)}-{min(g.home_score, g.away_score)}",
            "week": g.week,
            "opponent_rating": round(faced[(g.game_id, opp)], 1),
        }
        if margin > 0:
            wins.append(entry)
        elif margin < 0:
            losses.append(entry)
    return {
        "best_win": max(wins, key=lambda e: e["opponent_rating"], default=None),
        "worst_loss": min(losses, key=lambda e: e["opponent_rating"], default=None),
    }


def remaining_schedule(team: str, upcoming: list[Game], ratings: dict[str, float], hfa: float) -> float | None:
    """Average opponent strength still to play, in points, counting where the
    game is played (a road game against an average team is +hfa harder)."""
    games = [g for g in upcoming if team in (g.home, g.away) and g.game_type == "REG"]
    if not games:
        return None
    return sum(ratings[_opponent(team, g)] - _edge(team, g, hfa) for g in games) / len(games)
