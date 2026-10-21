"""Command line: python -m powerrank {backtest,fit,rank}."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import polars as pl

from powerrank import backtest, injuries, preseason
from powerrank.data import Game, load_games
from powerrank.models import MODELS, RatingModel
from powerrank.models.epa import (
    WEIGHTS_PATH,
    EpaModel,
    TeamGameStats,
    cross_validated_mae,
    fit_weights,
    load_weights,
)
from powerrank.models.qb import QbTracker
from powerrank.pbp import FIRST_SEASON, load_qb_games, load_team_games

DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "output" / "rankings.json"

# Preseason projections need two earlier seasons of play-by-play.
PRESEASON_FIRST_SEASON = FIRST_SEASON + 2

# A game this many days past its date with no score is treated as cancelled,
# so it can't hold a week open forever.
CANCELLED_AFTER_DAYS = 3


def season_range(text: str) -> range:
    start, _, end = text.partition("-")
    return range(int(start), int(end or start) + 1)


@dataclass
class Inputs:
    """Everything the epa model reads besides the schedule."""

    stats: TeamGameStats
    qb_games: pl.DataFrame
    injuries: dict[tuple[int, int, str], float]


def load_inputs(games: list[Game], refresh: bool) -> Inputs:
    last = max(g.season for g in games if g.played)
    seasons = range(FIRST_SEASON, last + 1)
    return Inputs(
        stats=TeamGameStats(load_team_games(seasons, refresh=refresh)),
        qb_games=load_qb_games(seasons, refresh=refresh),
        injuries=injuries.load_index(last, refresh=refresh),
    )


def build_epa(
    games: list[Game], inputs: Inputs, fit_seasons: range | None = None
) -> tuple[EpaModel, dict[str, float], list]:
    """The epa model with preseason projections, QB ratings and injuries.

    With `fit_seasons`, refits the projections and weights on just those
    seasons (and earlier), so a backtest never scores games anything was fitted
    on, and also returns the weights and the per-game rows they were fitted on.
    Otherwise uses the saved fit."""
    if fit_seasons is None:
        pre = preseason.build(inputs.stats, games, QbTracker(inputs.qb_games), saved=load_weights()["preseason"])
    else:
        pre = preseason.build(
            inputs.stats, games, QbTracker(inputs.qb_games), seasons=range(PRESEASON_FIRST_SEASON, fit_seasons.stop)
        )

    def make(weights: dict[str, float] | None) -> EpaModel:
        return EpaModel(
            stats=inputs.stats, weights=weights, preseason=pre, qbs=QbTracker(inputs.qb_games), injuries=inputs.injuries
        )

    if fit_seasons is None:
        return make(None), {}, []
    weights, rows = fit_weights(make({}), games, fit_seasons)
    return make(weights), weights, rows


def build_model(name: str, games: list[Game], inputs: Inputs | None, fit_seasons: range | None = None) -> RatingModel:
    if name != "epa":
        return MODELS[name]()
    return build_epa(games, inputs, fit_seasons)[0]


def cmd_backtest(args: argparse.Namespace, games: list[Game]) -> None:
    names = [args.model] if args.model else list(MODELS)
    inputs = load_inputs(games, args.refresh) if "epa" in names else None
    for label, seasons in (("train", args.train), ("test", args.test)):
        print(f"\n{label} {seasons.start}-{seasons.stop - 1}")
        print(f"{'':8}{backtest.SCORES_HEADER}")
        market = None
        for name in names:
            model = build_model(name, games, inputs, fit_seasons=args.train)
            predictions = backtest.run(model, games, through_season=seasons.stop - 1)
            model_scores, market = backtest.evaluate(predictions, seasons)
            print(f"{name:8}{model_scores.row()}")
        print(f"{'market':8}{market.row()}")


def cmd_fit(args: argparse.Namespace, games: list[Game]) -> None:
    seasons = args.seasons or range(2006, max(g.season for g in games if g.played))
    model, weights, rows = build_epa(games, load_inputs(games, args.refresh), fit_seasons=seasons)
    pre = model.preseason
    params = {
        "prior_plays": model.prior_plays,
        "prior_scale": model.prior_scale,
        "carryover": model.carryover,
        "garbage_weight": model.garbage_weight,
        "st_prior_games": model.st_prior_games,
        "half_life": model.half_life,
        "neutralize_fumbles": model.neutralize_fumbles,
        "qb": {
            "half_life_weeks": model.qbs.half_life_weeks,
            "prior_plays": model.qbs.prior_plays,
            "prior_mean": model.qbs.prior_mean,
        },
    }
    WEIGHTS_PATH.write_text(
        json.dumps(
            {
                "fitted_on": f"{seasons.start}-{seasons.stop - 1}",
                "games": len(rows),
                "cross_validated_mae": round(cross_validated_mae(rows), 3),
                "params": params,
                "weights": {f: round(w, 4) for f, w in weights.items()},
                "preseason": pre.to_json(),
            },
            indent=2,
        )
        + "\n"
    )
    for f, w in weights.items():
        print(f"  {f:10} {w:+8.2f}")
    print(f"wrote {WEIGHTS_PATH}")


def completed_through(season_games: list[Game], today: date) -> int | None:
    """Last week such that it and every earlier week are fully played."""
    last = None
    for week in sorted({g.week for g in season_games}):
        for g in season_games:
            if g.week != week or g.played:
                continue
            gameday = date.fromisoformat(g.gameday) if g.gameday else None
            if gameday is None or today - gameday <= timedelta(days=CANCELLED_AFTER_DAYS):
                return last
        last = week
    return last


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


def cmd_rank(args: argparse.Namespace, games: list[Game]) -> None:
    season = args.season or max(g.season for g in games if g.played)
    season_games = [g for g in games if g.season == season]
    through = completed_through(season_games, date.today())
    if through is None:
        sys.exit(f"no completed weeks in {season} yet")

    # Only full weeks count, so a Thursday game never sneaks into a ranking.
    history = [g for g in games if g.season < season or (g.season == season and g.week <= through)]
    inputs = load_inputs(games, args.refresh) if args.model == "epa" else None
    model = build_model(args.model, games, inputs)
    backtest.run(model, history)

    played = [g for g in season_games if g.week <= through and g.played]
    if isinstance(model, EpaModel):
        missing = [gid for gid in model.missing_games if gid.startswith(f"{season}_")]
        if missing:
            sys.exit(f"play-by-play not published yet for: {', '.join(missing)}")

    upcoming = [g for g in season_games if g.week > through]
    next_week = min((g.week for g in upcoming), default=None)
    qb_names = {qb: name for g in games for qb, name in ((g.home_qb, g.home_qb_name), (g.away_qb, g.away_qb_name)) if qb}
    teams = []
    for team, rating in model.ratings().items():
        entry = {"team": team, "rating": round(rating, 2), **record(team, played)}
        if isinstance(model, EpaModel):
            entry.update(this_week(model, team, upcoming, qb_names))
            entry["breakdown"] = {unit: round(v, 2) for unit, v in model.breakdown(team).items()} | {
                "quarterback": entry.pop("qb_points"),
                "injuries": entry.pop("injury_points"),
            }
            entry["preseason_rating"] = round(model.preseason_rating(team), 2)
            if model.preseason is not None:
                info = model.preseason.offseason.get((season, team), {})
                entry["offseason"] = {k: bool(info.get(k)) for k in ("qb_changed", "veteran_qb", "coach_changed")}
        teams.append(entry)
    teams.sort(key=lambda t: -t["rating"])
    for i, t in enumerate(teams):
        t["rank"] = i + 1
    matchups = [
        {
            "home": g.home,
            "away": g.away,
            "neutral": g.neutral,
            "model_line": round(model.predict(g), 1),
            "market_line": g.spread_line,
        }
        for g in season_games
        if g.week == next_week
    ]

    print(f"{season} through week {through} ({model.name}, home field {model.hfa.value:+.1f})\n")
    has_breakdown = "breakdown" in teams[0]
    print(
        " #  team  rating  record  diff"
        + ("  healthy    off    def     st  score     qb    inj  starter" if has_breakdown else "")
    )
    for t in teams:
        rec = f"{t['wins']}-{t['losses']}" + (f"-{t['ties']}" if t["ties"] else "")
        line = f"{t['rank']:>2}  {t['team']:<4} {t['rating']:+7.1f}  {rec:>6}  {t['point_diff']:+4d}"
        if has_breakdown:
            b = t["breakdown"]
            line += (
                f"  {t['healthy_rating']:+7.1f}  {b['offense']:+5.1f}  {b['defense']:+5.1f}  {b['special_teams']:+5.1f}"
                f"  {b['scoring']:+5.1f}  {b['quarterback']:+5.1f}  {b['injuries']:+5.1f}  {t['starter'] or ''}"
            )
        print(line)
    if matchups:
        print(f"\nweek {next_week} (home team's expected margin)")
        for m in matchups:
            market = "n/a" if m["market_line"] is None else f"{m['market_line']:+.1f}"
            print(f"  {m['away']:>3} @ {m['home']:<3}  model {m['model_line']:+5.1f}  market {market:>5}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "season": season,
                "through_week": through,
                "model": model.name,
                "home_field": round(model.hfa.value, 2),
                "teams": teams,
                "next_week": {"week": next_week, "games": matchups},
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote {args.out}")


def this_week(model: EpaModel, team: str, upcoming: list[Game], qb_names: dict[str, str]) -> dict:
    """Strength for the team's next game (projected starter, injury report)
    and when healthy (its main QB, nobody ruled out). If the next starter
    isn't projected yet (e.g. after a bye), assume the main QB."""
    primary = model.primary_qb(team)
    healthy = model.rating_with(team, primary)
    game = next((g for g in upcoming if team in (g.home, g.away)), None)
    if game is None:
        return {
            "rating": round(healthy, 2), "healthy_rating": round(healthy, 2),
            "starter": qb_names.get(primary), "qb_points": 0.0, "injury_points": 0.0,
        }
    qb, name = (game.home_qb, game.home_qb_name) if game.home == team else (game.away_qb, game.away_qb_name)
    if qb is None:
        qb, name = primary, qb_names.get(primary)
    with_qb = model.rating_with(team, qb)
    injury = model.injury_cost(game.season, game.week, team)
    return {
        "rating": round(with_qb + injury, 2),
        "healthy_rating": round(healthy, 2),
        "starter": name,
        "qb_points": round(with_qb - model.ratings()[team], 2),
        "injury_points": round(injury, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="powerrank")
    parser.add_argument("--refresh", action="store_true", help="re-download data instead of using the cache")
    sub = parser.add_subparsers(dest="command", required=True)

    bt = sub.add_parser("backtest", help="score models week by week against history and the closing line")
    bt.add_argument("--model", choices=list(MODELS))
    bt.add_argument("--train", type=season_range, default=season_range("2006-2015"))
    bt.add_argument("--test", type=season_range, default=season_range("2016-2025"))

    ft = sub.add_parser("fit", help="fit the epa model's weights and save them")
    ft.add_argument("--seasons", type=season_range, help="default: 2006 through last season")

    rk = sub.add_parser("rank", help="current ratings")
    rk.add_argument("--model", choices=list(MODELS), default="epa")
    rk.add_argument("--season", type=int)
    rk.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)

    args = parser.parse_args()
    games = load_games(refresh=args.refresh)
    {"backtest": cmd_backtest, "fit": cmd_fit, "rank": cmd_rank}[args.command](args, games)


if __name__ == "__main__":
    main()
