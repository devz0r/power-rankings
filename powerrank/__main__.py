"""Command line: python -m powerrank {backtest,fit,rank}."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from powerrank import backtest, preseason
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
from powerrank.pbp import FIRST_SEASON, load_team_games

DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "output" / "rankings.json"

# Preseason projections need two earlier seasons of play-by-play.
PRESEASON_FIRST_SEASON = FIRST_SEASON + 2

# A game this many days past its date with no score is treated as cancelled,
# so it can't hold a week open forever.
CANCELLED_AFTER_DAYS = 3


def season_range(text: str) -> range:
    start, _, end = text.partition("-")
    return range(int(start), int(end or start) + 1)


def load_stats(games: list[Game], refresh: bool) -> TeamGameStats:
    last = max(g.season for g in games if g.played)
    return TeamGameStats(load_team_games(range(FIRST_SEASON, last + 1), refresh=refresh))


def build_model(name: str, games: list[Game], stats: TeamGameStats | None, fit_seasons: range | None = None) -> RatingModel:
    """A fresh model. For epa, `fit_seasons` refits its preseason projections
    and weights on just those seasons (and earlier), so a backtest never scores
    games anything was fitted on. Otherwise epa uses the saved fit."""
    if name != "epa":
        return MODELS[name]()
    if fit_seasons is None:
        saved = load_weights()
        pre = preseason.build(stats, games, saved=saved["preseason"])
        return EpaModel(stats=stats, preseason=pre)
    pre = preseason.build(stats, games, seasons=range(PRESEASON_FIRST_SEASON, fit_seasons.stop))
    weights, _ = fit_weights(EpaModel(stats=stats, weights={}, preseason=pre), games, fit_seasons)
    return EpaModel(stats=stats, weights=weights, preseason=pre)


def cmd_backtest(args: argparse.Namespace, games: list[Game]) -> None:
    names = [args.model] if args.model else list(MODELS)
    stats = load_stats(games, args.refresh) if "epa" in names else None
    for label, seasons in (("train", args.train), ("test", args.test)):
        print(f"\n{label} {seasons.start}-{seasons.stop - 1}")
        print(f"{'':8}{backtest.SCORES_HEADER}")
        market = None
        for name in names:
            model = build_model(name, games, stats, fit_seasons=args.train)
            predictions = backtest.run(model, games, through_season=seasons.stop - 1)
            model_scores, market = backtest.evaluate(predictions, seasons)
            print(f"{name:8}{model_scores.row()}")
        print(f"{'market':8}{market.row()}")


def cmd_fit(args: argparse.Namespace, games: list[Game]) -> None:
    seasons = args.seasons or range(2006, max(g.season for g in games if g.played))
    stats = load_stats(games, args.refresh)
    pre = preseason.build(stats, games, seasons=range(PRESEASON_FIRST_SEASON, seasons.stop))
    model = EpaModel(stats=stats, weights={}, preseason=pre)
    weights, rows = fit_weights(model, games, seasons)
    params = {
        "prior_plays": model.prior_plays,
        "prior_scale": model.prior_scale,
        "carryover": model.carryover,
        "garbage_weight": model.garbage_weight,
        "st_prior_games": model.st_prior_games,
        "half_life": model.half_life,
        "neutralize_fumbles": model.neutralize_fumbles,
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
    stats = load_stats(games, args.refresh) if args.model == "epa" else None
    model = build_model(args.model, games, stats)
    backtest.run(model, history)

    played = [g for g in season_games if g.week <= through and g.played]
    if isinstance(model, EpaModel):
        missing = [gid for gid in model.missing_games if gid.startswith(f"{season}_")]
        if missing:
            sys.exit(f"play-by-play not published yet for: {', '.join(missing)}")

    next_week = min((g.week for g in season_games if g.week > through), default=None)
    ratings = sorted(model.ratings().items(), key=lambda item: -item[1])
    teams = []
    for i, (team, rating) in enumerate(ratings):
        entry = {"rank": i + 1, "team": team, "rating": round(rating, 2), **record(team, played)}
        if isinstance(model, EpaModel):
            entry["breakdown"] = {unit: round(v, 2) for unit, v in model.breakdown(team).items()}
            entry["preseason_rating"] = round(model.preseason_rating(team), 2)
            if model.preseason is not None:
                info = model.preseason.offseason.get((season, team), {})
                entry["offseason"] = {k: bool(info.get(k)) for k in ("qb_changed", "veteran_qb", "coach_changed")}
        teams.append(entry)
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
    print(" #  team  rating  record  diff" + ("    off    def     st  score" if has_breakdown else ""))
    for t in teams:
        rec = f"{t['wins']}-{t['losses']}" + (f"-{t['ties']}" if t["ties"] else "")
        line = f"{t['rank']:>2}  {t['team']:<4} {t['rating']:+7.1f}  {rec:>6}  {t['point_diff']:+4d}"
        if has_breakdown:
            b = t["breakdown"]
            line += f"  {b['offense']:+5.1f}  {b['defense']:+5.1f}  {b['special_teams']:+5.1f}  {b['scoring']:+5.1f}"
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
