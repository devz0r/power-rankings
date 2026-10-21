"""Command line: python -m powerrank {backtest,rank}."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from powerrank import backtest
from powerrank.data import Game, load_games
from powerrank.models import MODELS

DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "output" / "rankings.json"


def season_range(text: str) -> range:
    start, _, end = text.partition("-")
    return range(int(start), int(end or start) + 1)


def cmd_backtest(args: argparse.Namespace, games: list[Game]) -> None:
    names = [args.model] if args.model else list(MODELS)
    for label, seasons in (("train", args.train), ("test", args.test)):
        print(f"\n{label} {seasons.start}-{seasons.stop - 1}")
        print(f"{'':8}{backtest.SCORES_HEADER}")
        market = None
        for name in names:
            predictions = backtest.run(MODELS[name](), games, through_season=seasons.stop - 1)
            model_scores, market = backtest.evaluate(predictions, seasons)
            print(f"{name:8}{model_scores.row()}")
        print(f"{'market':8}{market.row()}")


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
    model = MODELS[args.model]()
    backtest.run(model, games, through_season=season)

    season_games = [g for g in games if g.season == season]
    played = [g for g in season_games if g.played]
    upcoming = [g for g in season_games if not g.played]
    next_week = min((g.week for g in upcoming), default=None)

    ratings = sorted(model.ratings().items(), key=lambda item: -item[1])
    teams = [
        {"rank": i + 1, "team": team, "rating": round(rating, 2), **record(team, played)}
        for i, (team, rating) in enumerate(ratings)
    ]
    matchups = [
        {
            "home": g.home,
            "away": g.away,
            "neutral": g.neutral,
            "model_line": round(model.predict(g), 1),
            "market_line": g.spread_line,
        }
        for g in upcoming
        if g.week == next_week
    ]

    print(f"{season} through week {max(g.week for g in played)} ({model.name}, home field {model.hfa.value:+.1f})\n")
    print(" #  team  rating  record  diff")
    for t in teams:
        rec = f"{t['wins']}-{t['losses']}" + (f"-{t['ties']}" if t["ties"] else "")
        print(f"{t['rank']:>2}  {t['team']:<4} {t['rating']:+7.1f}  {rec:>6}  {t['point_diff']:+4d}")
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
                "through_week": max(g.week for g in played),
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

    rk = sub.add_parser("rank", help="current ratings")
    rk.add_argument("--model", choices=list(MODELS), default="ridge")
    rk.add_argument("--season", type=int)
    rk.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)

    args = parser.parse_args()
    games = load_games(refresh=args.refresh)
    {"backtest": cmd_backtest, "rank": cmd_rank}[args.command](args, games)


if __name__ == "__main__":
    main()
