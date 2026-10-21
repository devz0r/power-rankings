import polars as pl
import pytest

from powerrank import backtest
from powerrank.data import Game
from powerrank.injuries import build_index
from powerrank.models.epa import EpaModel, TeamGameStats
from powerrank.models.qb import QbTracker


def game(i: int, week: int, home_qb: str, away_qb: str = "opp_qb") -> Game:
    return Game(
        game_id=f"g{i}", season=2020, week=week, game_type="REG", home="AAA", away="BBB", neutral=False,
        gameday=None, home_score=20, away_score=17, spread_line=0.0, home_qb=home_qb, away_qb=away_qb,
    )


def qb_row(g: Game, team: str, qb: str, epa_per_play: float, plays: int = 40) -> dict:
    defense = "BBB" if team == "AAA" else "AAA"
    return {
        "game_id": g.game_id, "season": g.season, "week": g.week, "team": team, "defense": defense,
        "qb": qb, "plays": plays, "epa": epa_per_play * plays, "fumbles": 0, "fumble_epa": 0.0,
    }


def team_rows(g: Game) -> list[dict]:
    rows = []
    for off, dfn in (("AAA", "BBB"), ("BBB", "AAA")):
        rows.append({
            "game_id": g.game_id, "season": g.season, "week": g.week, "offense": off, "defense": dfn,
            "offense_home": off == "AAA", "kind": "pass", "garbage": False, "plays": 40, "epa": 0.0,
            "fumbles": 0, "fumble_epa": 0.0, "interceptions": 0, "interception_epa": 0.0,
        })
    return rows


def tracker(rows: list[dict], **kwargs) -> QbTracker:
    return QbTracker(pl.DataFrame(rows), **kwargs)


def test_unknown_qb_gets_the_prior():
    qbs = tracker([qb_row(game(0, 1, "x"), "AAA", "x", 0.0)], prior_mean=-0.2)
    assert qbs.rating("never_played") == pytest.approx(-0.2)


def test_better_qb_rates_higher_and_shrinks_with_few_plays():
    games = [game(i, i + 1, "star") for i in range(10)]
    rows = [qb_row(g, "AAA", "star", 0.3) for g in games] + [qb_row(g, "BBB", "scrub", -0.1) for g in games]
    rows.append(qb_row(games[0], "AAA", "one_game_wonder", 0.6, plays=10))
    qbs = tracker(rows, prior_plays=100, prior_mean=-0.2)
    for g in games:
        qbs.update_week([g], defense_allowed={}, fumble_epa=-2.0)
    assert qbs.rating("star") > qbs.rating("scrub")
    # 10 great plays don't outweigh a 100-play prior.
    assert qbs.rating("one_game_wonder") < qbs.rating("star")


def test_old_plays_fade():
    g1, g2 = game(1, 1, "x"), game(2, 2, "x")
    rows = [qb_row(g1, "AAA", "x", 0.5), qb_row(g2, "AAA", "x", -0.5)]
    fast, slow = tracker(rows, half_life_weeks=1, prior_plays=1), tracker(rows, half_life_weeks=100, prior_plays=1)
    for qbs in (fast, slow):
        qbs.update_week([g1], {}, -2.0)
        qbs.update_week([g2], {}, -2.0)
    # With a short half-life the recent bad game dominates.
    assert fast.rating("x") < slow.rating("x")


def test_backup_starter_lowers_the_prediction():
    games = [game(i, i + 1, "starter") for i in range(6)] + [game(6, 7, "backup")]
    qb_rows, rows = [], []
    for g in games[:6]:
        qb_rows += [qb_row(g, "AAA", "starter", 0.3), qb_row(g, "BBB", "opp_qb", 0.0)]
        rows += team_rows(g)
    model = EpaModel(
        stats=TeamGameStats(pl.DataFrame(rows)), weights={"qb": 20.0}, prior_plays=10,
        qbs=tracker(qb_rows, prior_plays=50, prior_mean=-0.2),
    )
    backtest.run(model, games[:6])
    assert model.qb_adjustment("AAA", "starter") == pytest.approx(0.0, abs=0.02)
    assert model.qb_adjustment("AAA", "backup") < -0.2
    assert model.predict(games[6]) < model.predict(games[5]) - 4
    assert model.primary_qb("AAA") == "starter"


def test_injury_index_counts_recent_snap_share():
    snaps = pl.DataFrame({
        "season": [2020] * 6, "week": [1, 2, 3, 1, 2, 3], "team": ["AAA"] * 6,
        "pfr_player_id": ["starter"] * 3 + ["long_out"] * 3,
        "offense_pct": [1.0, 0.9, 0.8, 0.0, 0.0, 0.0], "defense_pct": [0.0] * 6,
    })
    injuries = pl.DataFrame({
        "season": [2020] * 3, "week": [4] * 3, "team": ["AAA"] * 3,
        "gsis_id": ["g_starter", "g_long_out", "g_qb"], "position": ["T", "WR", "QB"],
        "report_status": ["Out", "Out", "Out"],
    })
    players = pl.DataFrame({"gsis_id": ["g_starter", "g_long_out", "g_qb"], "pfr_id": ["starter", "long_out", "qb"]})
    index = build_index(injuries, snaps, players)
    # The starter averaged 90% of snaps; the long-term absence is already priced in; QBs are skipped.
    assert index[(2020, 4, "AAA")] == pytest.approx(0.9)
