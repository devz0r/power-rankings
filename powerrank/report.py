"""Everything the rankings page shows, for one completed week."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from powerrank import backtest, market, resume, simulate, tiers, uncertainty
from powerrank.data import Game
from powerrank.models.base import win_probability
from powerrank.models.epa import EpaModel
from powerrank.teams import Team

# Ranges shown are 80% intervals.
RANGE_Z = 1.2816
# Strength of record compares against a typical playoff team: the average
# rating of the top 14.
SOR_REFERENCE_TEAMS = 14
N_DRAWS = 10_000


def this_week(model: EpaModel, team: str, upcoming: list[Game], qb_names: dict[str, str]) -> dict:
    """Strength for the team's next game (projected starter, injury report)
    and when healthy (its main QB, nobody ruled out). If the next starter
    isn't projected yet (e.g. after a bye), assume the main QB."""
    primary = model.primary_qb(team)
    healthy = model.rating_with(team, primary)
    game = next((g for g in upcoming if team in (g.home, g.away)), None)
    if game is None:
        return {
            "rating": healthy, "healthy_rating": healthy,
            "starter": qb_names.get(primary), "qb_points": 0.0, "injury_points": 0.0,
        }
    qb, name = (game.home_qb, game.home_qb_name) if game.home == team else (game.away_qb, game.away_qb_name)
    if qb is None:
        qb, name = primary, qb_names.get(primary)
    with_qb = model.rating_with(team, qb)
    injury = model.injury_cost(game.season, game.week, team)
    return {
        "rating": with_qb + injury,
        "healthy_rating": healthy,
        "starter": name,
        "qb_points": with_qb - model.ratings()[team],
        "injury_points": injury,
    }


def ranks(values: dict[str, float], reverse: bool = True) -> dict[str, int]:
    order = sorted(values, key=values.get, reverse=reverse)
    return {t: i + 1 for i, t in enumerate(order)}


def build(
    model: EpaModel,
    games: list[Game],
    season: int,
    through: int,
    teams: dict[str, Team],
    calibration: dict,
) -> dict:
    """Replay history through `through` and assemble the full report."""
    season_games = [g for g in games if g.season == season]
    history = [g for g in games if g.season < season or (g.season == season and g.week <= through)]
    qb_names = {qb: name for g in games for qb, name in ((g.home_qb, g.home_qb_name), (g.away_qb, g.away_qb_name)) if qb}
    season_teams = {t for g in season_games for t in (g.home, g.away)}

    # Last week's ratings first, then absorb the latest week.
    backtest.run(model, [g for g in history if not (g.season == season and g.week == through)])
    if model._season != season:
        model.start_season(season, season_teams)
    last_upcoming = [g for g in season_games if g.week >= through]
    previous = {t: this_week(model, t, last_upcoming, qb_names)["rating"] for t in season_teams}
    model.update_week([g for g in season_games if g.week == through and g.played])

    played = [g for g in season_games if g.played and g.week <= through]
    upcoming = [g for g in season_games if not g.played and g.week > through]
    next_week = min((g.week for g in upcoming), default=None)
    hfa = model.hfa.value

    now = {t: this_week(model, t, upcoming, qb_names) for t in season_teams}
    rating = {t: v["rating"] for t, v in now.items()}
    healthy = {t: v["healthy_rating"] for t, v in now.items()}
    sd = uncertainty.rating_sd(calibration, next_week or through + 1)

    # Résumé, judged against current healthy strength.
    reference = float(np.mean(sorted(healthy.values(), reverse=True)[:SOR_REFERENCE_TEAMS]))
    sor = {t: resume.strength_of_record(t, played, healthy, hfa, reference) for t in season_teams}
    remaining = {t: resume.remaining_schedule(t, upcoming, healthy, hfa) for t in season_teams}

    # How sure: chance each team is truly the best / top five.
    rng = np.random.default_rng(season * 100 + through)
    order = sorted(season_teams)
    draws = np.array([rating[t] for t in order]) + rng.normal(0.0, sd, (N_DRAWS, len(order)))
    place = (-draws).argsort(axis=1).argsort(axis=1)
    chance_best = dict(zip(order, (place == 0).mean(axis=0).tolist()))
    chance_top5 = dict(zip(order, (place < 5).mean(axis=0).tolist()))

    # Rest of the season: next week's games with projected starters and
    # injuries, later games with each team at full health.
    sim_games = []
    for g in upcoming:
        if g.game_type != "REG":
            continue
        mean = model.predict(g) if g.week == next_week else healthy[g.home] - healthy[g.away] + (0 if g.neutral else hfa)
        sim_games.append((g, mean))
    odds = simulate.simulate(
        {t: teams[t] for t in season_teams}, played, sim_games, healthy, sd, calibration["game_sd"], hfa,
        seed=season * 100 + through,
    )

    vegas, vegas_hfa = market.implied_ratings(season_games, next_week or through)
    tier = tiers.assign(rating)
    rank, prev_rank = ranks(rating), ranks(previous)
    sor_rank = ranks(sor, reverse=False)
    sos_rank = ranks({t: v for t, v in remaining.items() if v is not None})

    entries = []
    for t in sorted(season_teams, key=rank.get):
        info = teams[t]
        rec = resume.record(t, played)
        offseason = model.preseason.offseason.get((season, t), {}) if model.preseason else {}
        entries.append({
            "rank": rank[t],
            "previous_rank": prev_rank[t],
            "rank_change": prev_rank[t] - rank[t],
            "team": t,
            "name": info.name,
            "nickname": info.nickname,
            "conference": info.conference,
            "division": info.division,
            "color": info.color,
            "color2": info.color2,
            "logo": info.logo,
            "tier": tier[t],
            "rating": round(rating[t], 2),
            "rating_change": round(rating[t] - previous[t], 2),
            "range": [round(rating[t] - RANGE_Z * sd, 1), round(rating[t] + RANGE_Z * sd, 1)],
            "healthy_rating": round(healthy[t], 2),
            "starter": now[t]["starter"],
            "breakdown": {k: round(v, 2) for k, v in model.breakdown(t).items()} | {
                "quarterback": round(now[t]["qb_points"], 2),
                "injuries": round(now[t]["injury_points"], 2),
            },
            "preseason_rating": round(model.preseason_rating(t), 2),
            "offseason": {k: bool(offseason.get(k)) for k in ("qb_changed", "veteran_qb", "coach_changed")},
            **rec,
            "strength_of_record": round(sor[t], 3),
            "strength_of_record_rank": sor_rank[t],
            **resume.notable_results(t, played, healthy),
            "remaining_schedule": None if remaining[t] is None else round(remaining[t], 2),
            "remaining_schedule_rank": sos_rank.get(t),
            "chance_best": round(chance_best[t], 3),
            "chance_top5": round(chance_top5[t], 3),
            "odds": {
                "projected_wins": round(odds.mean_wins[t], 1),
                "make_playoffs": round(odds.playoffs[t], 3),
                "win_division": round(odds.division[t], 3),
                "top_seed": round(odds.bye[t], 3),
                "win_conference": round(odds.conference[t], 3),
                "win_super_bowl": round(odds.super_bowl[t], 3),
            },
            "vegas_rating": round(vegas[t], 2) if t in vegas else None,
            "vegas_difference": round(rating[t] - vegas[t], 2) if t in vegas else None,
        })

    matchups = []
    for g in season_games:
        if g.week != next_week:
            continue
        line = model.predict(g)
        matchups.append({
            "home": g.home,
            "away": g.away,
            "neutral": g.neutral,
            "gameday": g.gameday,
            "home_starter": g.home_qb_name,
            "away_starter": g.away_qb_name,
            "model_line": round(line, 1),
            "home_win_probability": round(win_probability(line, calibration["game_sd"]), 3),
            "market_line": g.spread_line,
        })

    return {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "season": season,
        "through_week": through,
        "next_week": next_week,
        "model": model.name,
        "home_field": round(hfa, 2),
        "vegas_home_field": round(vegas_hfa, 2),
        "rating_sd": sd,
        "game_sd": calibration["game_sd"],
        "strength_of_record_reference": round(reference, 2),
        "teams": entries,
        "games": matchups,
    }
