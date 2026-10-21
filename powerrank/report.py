"""Everything the rankings page shows, for one completed week."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from powerrank import backtest, market, notes, resume, simulate, tiers, uncertainty
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


def this_week(
    model: EpaModel,
    team: str,
    upcoming: list[Game],
    qb_names: dict[str, str],
    available=None,
    played: list[Game] | None = None,
) -> dict:
    """Strength for the team's next game (projected starter, injury report)
    and when healthy (its main QB, nobody ruled out). If the next starter
    isn't projected yet (e.g. after a bye), assume the main QB. A main QB the
    roster shows as unavailable (e.g. on injured reserve) doesn't count as
    the main QB; the active replacement does."""
    primary = model.primary_qb(team)
    game = next((g for g in upcoming if team in (g.home, g.away)), None)
    if available is not None and game is not None and available.is_out(game.season, game.week, team, primary):
        primary = available.replacement(game.season, game.week, team, played or []) or primary
        qb_names = qb_names | {primary: available.names.get(primary)}
    healthy = model.rating_with(team, primary)
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


def team_context(team: str, played: list[Game], upcoming: list[Game], rank: dict[str, int],
                 teams: dict[str, Team], model: EpaModel, through: int) -> dict:
    """Last game, streak and next game, for the team's weekly note."""
    mine = [g for g in played if team in (g.home, g.away)]
    ctx: dict = {}
    if mine and mine[-1].week == through:
        g = mine[-1]
        margin = g.home_margin if g.home == team else -g.home_margin
        opp = g.away if g.home == team else g.home
        score = f"{max(g.home_score, g.away_score)}-{min(g.home_score, g.away_score)}"
        word = "win over" if margin > 0 else "loss to" if margin < 0 else "tie with"
        place = "at home" if g.home == team and not g.neutral else "on the road" if g.away == team else "on a neutral field"
        ctx["last_game"] = f"a {score} {word} the {teams[opp].nickname} (ranked #{rank[opp]}) {place}"
    results = ["W" if (g.home_margin if g.home == team else -g.home_margin) > 0 else
               "L" if (g.home_margin if g.home == team else -g.home_margin) < 0 else "T" for g in mine]
    if results:
        run = 0
        for r in reversed(results):
            if r == results[-1]:
                run += 1
            else:
                break
        if run >= 3 and results[-1] in "WL":
            ctx["streak"] = f"{'won' if results[-1] == 'W' else 'lost'} {run} straight"
    nxt = next((g for g in upcoming if team in (g.home, g.away)), None)
    if nxt is not None:
        home = nxt.home == team
        opp = nxt.away if home else nxt.home
        line = model.predict(nxt) * (1 if home else -1)
        where = "at home" if home and not nxt.neutral else "on the road" if not home else "on a neutral field"
        side = "favored" if line > 0 else "underdog"
        ctx["next_game"] = f"week {nxt.week}, {where} against the {teams[opp].nickname} (ranked #{rank[opp]}), model has them as {side} by {abs(line):.1f}"
    return ctx


def build(
    model: EpaModel,
    games: list[Game],
    season: int,
    through: int,
    teams: dict[str, Team],
    calibration: dict,
    previous_notes: dict[str, dict] | None = None,
    client=None,
    available=None,
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
    last_week = {t: this_week(model, t, last_upcoming, qb_names) for t in season_teams}
    previous = {t: v["rating"] for t, v in last_week.items()}
    previous_breakdown = {
        t: model.breakdown(t) | {"quarterback": last_week[t]["qb_points"], "injuries": last_week[t]["injury_points"]}
        for t in season_teams
    }
    model.update_week([g for g in season_games if g.week == through and g.played])

    played = [g for g in season_games if g.played and g.week <= through]
    upcoming = [g for g in season_games if not g.played and g.week > through]
    next_week = min((g.week for g in upcoming), default=None)
    hfa = model.hfa.value

    now = {t: this_week(model, t, upcoming, qb_names, available, played) for t in season_teams}
    rating = {t: v["rating"] for t, v in now.items()}
    healthy = {t: v["healthy_rating"] for t, v in now.items()}
    sd = uncertainty.rating_sd(calibration, next_week or through + 1)

    # Résumé, judged against current healthy strength.
    reference = float(np.mean(sorted(healthy.values(), reverse=True)[:SOR_REFERENCE_TEAMS]))
    # Each past opponent as fielded that day: its QB and that week's injuries.
    faced = {
        (g.game_id, t): model.rating_with(t, qb) + model.injury_cost(g.season, g.week, t)
        for g in played
        for t, qb in ((g.home, g.home_qb), (g.away, g.away_qb))
    }
    sor = {t: resume.strength_of_record(t, played, faced, hfa, reference) for t in season_teams}
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
            "abbr": info.display_abbr,
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
            **resume.notable_results(t, played, faced, teams),
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

    contexts = {
        t: team_context(t, played, upcoming, rank, teams, model, through) | {"previous_breakdown": previous_breakdown[t]}
        for t in season_teams
    }
    note_counts = notes.attach(entries, contexts, previous_notes or {}, client)

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
        "note_counts": note_counts,
    }
