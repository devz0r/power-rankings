"""Command line: python -m powerrank {backtest,fit,rank}."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from powerrank import backtest, injuries, notes, preseason, report, rosters, uncertainty
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
from powerrank.teams import load_teams

DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "site" / "rankings.json"

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
    calibration = uncertainty.calibrate(model, games, seasons)
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
                "uncertainty": calibration,
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


def cmd_rank(args: argparse.Namespace, games: list[Game]) -> None:
    season = args.season or max(g.season for g in games if g.played)
    season_games = [g for g in games if g.season == season]
    through = completed_through(season_games, date.today())
    if through is None:
        sys.exit(f"no completed weeks in {season} yet")

    model = build_model("epa", games, load_inputs(games, args.refresh))
    saved = load_weights()
    previous_notes = {}
    if args.out.exists():
        try:
            previous_notes = {t["team"]: t for t in json.loads(args.out.read_text()).get("teams", [])}
        except (ValueError, KeyError):
            pass
    client = None if args.no_ai else notes.make_client()
    try:
        available = rosters.QbAvailability(rosters.load_rosters(season, refresh=args.refresh))
    except Exception as exc:  # rosters only refine projections; rank without them
        print(f"rosters unavailable ({type(exc).__name__}: {exc}); using schedule's projected starters")
        available = None
    if available is not None:
        corrected = rosters.correct_starters(games, available)
        for old, new in zip(games, corrected):
            for side in ("home", "away"):
                if getattr(old, f"{side}_qb") not in (None, getattr(new, f"{side}_qb")):
                    print(f"week {new.week} {getattr(new, side)}: {getattr(old, f'{side}_qb_name')} isn't active, "
                          f"projecting {getattr(new, f'{side}_qb_name')}")
        games = corrected
    data = report.build(
        model, games, season, through, load_teams(refresh=args.refresh), saved["uncertainty"], previous_notes, client,
        available,
    )
    missing = [gid for gid in model.missing_games if gid.startswith(f"{season}_")]
    if missing:
        sys.exit(f"play-by-play not published yet for: {', '.join(missing)}")

    print(f"{season} through week {through} (home field {data['home_field']:+.1f}, ratings +/- {data['rating_sd']:.1f})\n")
    print("  #   chg  team tier rating     80% range  record  SOR  playoff    SB  vegas  starter")
    for t in data["teams"]:
        rec = f"{t['wins']}-{t['losses']}" + (f"-{t['ties']}" if t["ties"] else "")
        odds = t["odds"]
        vegas = "" if t["vegas_difference"] is None else f"{t['vegas_difference']:+5.1f}"
        print(
            f"{t['rank']:>3}  {t['rank_change']:+3d}  {t['team']:<4}  {t['tier']}  {t['rating']:+6.1f}  "
            f"[{t['range'][0]:+5.1f},{t['range'][1]:+5.1f}]  {rec:>6}  {t['strength_of_record_rank']:>3}  "
            f"{odds['make_playoffs']:6.0%}  {odds['win_super_bowl']:4.0%}  {vegas:>5}  {t['starter'] or ''}"
        )
    if data["games"]:
        print(f"\nweek {data['next_week']} (home team's expected margin)")
        for m in data["games"]:
            market = "n/a" if m["market_line"] is None else f"{m['market_line']:+.1f}"
            print(f"  {m['away']:>3} @ {m['home']:<3}  model {m['model_line']:+5.1f}  market {market:>5}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2) + "\n")
    counts = data["note_counts"]
    print(f"\nnotes: {counts['ai']} written by Claude, {counts['template']} from templates, {counts['reused']} reused")
    print(f"wrote {args.out}")


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
    rk.add_argument("--season", type=int)
    rk.add_argument("--no-ai", action="store_true", help="write the team notes from templates only")
    rk.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)

    args = parser.parse_args()
    games = load_games(refresh=args.refresh)
    {"backtest": cmd_backtest, "fit": cmd_fit, "rank": cmd_rank}[args.command](args, games)


if __name__ == "__main__":
    main()
