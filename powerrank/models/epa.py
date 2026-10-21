"""Opponent-adjusted efficiency model built on play-by-play EPA.

For each team it estimates, in EPA per play:
  off_pass / def_pass   dropback efficiency gained / allowed
  off_rush / def_rush   designed-run efficiency gained / allowed
  off_int / def_int     the interception share of dropback EPA (so the
                        weights can discount it as partly luck)
and, in EPA per game:
  st_kick               own field goals and extra points
  st_return             kickoff and punt units, net of the opponent's
plus `margin`, the score-based ridge rating, in points.

Each per-play component is a weighted ridge regression over this season's
games, solved for every offense and defense at once:
    play EPA = league mean + home edge + offense[o] + defense[d]
with each team pulled toward its preseason projection (powerrank.preseason),
held with a strength based on how accurate that projection has been
historically, so it fades as the season fills in. Without a projection, the
prior is a fraction of last season's estimate worth `prior_plays` plays.

Luck and garbage time:
  - Fumbles are charged at the league-average EPA of a fumble, so who
    recovered the ball doesn't count.
  - Plays outside 10-90% win probability count `garbage_weight` of a play.

A team's rating in points is a weighted sum of these features. The weights
are fitted by regressing game margins on pre-game feature differences over
the training seasons (see `fit`), and stored in epa_weights.json.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import polars as pl

from powerrank.data import Game
from powerrank.models.base import RatingModel
from powerrank.models.ridge import RidgeMarginModel

WEIGHTS_PATH = Path(__file__).with_name("epa_weights.json")

PLAY_COMPONENTS = ("pass", "rush", "int")
GAME_COMPONENTS = ("st_kick", "st_return")
FEATURES = (
    "off_pass", "def_pass", "off_rush", "def_rush", "off_int", "def_int",
    "st_kick", "st_return", "margin",
)
# Weights also cover game-specific adjustments, which depend on who plays
# rather than being team features: the QB starting, and the share of the
# usual lineup ruled out (see powerrank.injuries).
GAME_ADJUSTMENTS = ("qb", "injuries")
WEIGHT_NAMES = (*FEATURES, *GAME_ADJUSTMENTS)

# Which features make up each unit when a rating is broken down for display.
UNITS = {
    "offense": ("off_pass", "off_rush", "off_int"),
    "defense": ("def_pass", "def_rush", "def_int"),
    "special_teams": ("st_kick", "st_return"),
    "scoring": ("margin",),
}


def load_weights(path: Path = WEIGHTS_PATH) -> dict:
    return json.loads(path.read_text())


class TeamGameStats:
    """Aggregated play-by-play, indexed by game for fast weekly lookups."""

    def __init__(self, df: pl.DataFrame):
        self.by_game: dict[str, list[dict]] = defaultdict(list)
        for row in df.iter_rows(named=True):
            self.by_game[row["game_id"]].append(row)

        # League-average EPA of a fumble on offensive plays, per season.
        offense = df.filter(pl.col("kind").is_in(["pass", "rush"]))
        per_season = (
            offense.group_by("season")
            .agg(pl.col("fumble_epa").sum(), pl.col("fumbles").sum())
            .sort("season")
        )
        self._fumble_totals = {
            r["season"]: (r["fumble_epa"], r["fumbles"]) for r in per_season.iter_rows(named=True)
        }

    def fumble_epa(self, season: int) -> float:
        """Average EPA of a fumble in seasons before `season` (its own season if none)."""
        earlier = [v for s, v in self._fumble_totals.items() if s < season] or [
            self._fumble_totals.get(season, (-2.0, 1))
        ]
        epa = sum(e for e, _ in earlier)
        count = sum(n for _, n in earlier)
        return epa / count if count else -2.0


class EpaModel(RatingModel):
    name = "epa"

    def __init__(
        self,
        stats: TeamGameStats | None = None,
        weights: dict[str, float] | None = None,
        prior_plays: float | None = None,
        carryover: float | None = None,
        garbage_weight: float | None = None,
        st_prior_games: float | None = None,
        prior_scale: dict[str, float] | None = None,
        half_life: float | None = None,
        neutralize_fumbles: bool | None = None,
        preseason=None,
        qbs=None,
        injuries: dict | None = None,
    ):
        super().__init__()
        saved = load_weights() if WEIGHTS_PATH.exists() else {"params": {}, "weights": {}}
        params = saved["params"]
        self.weights = weights if weights is not None else saved["weights"]
        self.prior_plays = prior_plays if prior_plays is not None else params.get("prior_plays", 100.0)
        self.carryover = carryover if carryover is not None else params.get("carryover", 0.6)
        self.garbage_weight = garbage_weight if garbage_weight is not None else params.get("garbage_weight", 1.0)
        self.st_prior_games = st_prior_games if st_prior_games is not None else params.get("st_prior_games", 4.0)
        # Per-feature multiplier on prior_plays: noisier units get a stronger pull.
        self.prior_scale = prior_scale if prior_scale is not None else params.get("prior_scale", {})
        # Weeks after which a game counts half as much; None means no decay.
        self.half_life = half_life if half_life is not None else params.get("half_life")
        self.neutralize_fumbles = (
            neutralize_fumbles if neutralize_fumbles is not None else params.get("neutralize_fumbles", True)
        )
        if stats is None:
            from powerrank.pbp import FIRST_SEASON, load_team_games
            from datetime import date

            stats = TeamGameStats(load_team_games(range(FIRST_SEASON, date.today().year + 1)))
        self.stats = stats
        # A powerrank.preseason.Preseason. When set, it supplies each team's
        # starting point and how strongly to hold it, replacing `carryover`,
        # `prior_plays`, `prior_scale` and `st_prior_games`.
        self.preseason = preseason
        self._strength: dict[str, float] = {}
        # A powerrank.models.qb.QbTracker. When set, predictions adjust for
        # who starts at quarterback.
        self.qbs = qbs
        # Plays by each QB for each team this season, and the week-1 starter,
        # who carries the preseason prior's weight in the team's QB mix.
        self._team_qbs: dict[str, dict[str, float]] = {}
        self._qb_prior: dict[str, tuple[str, float]] = {}
        # (season, week, team) -> recent snap share of players ruled out.
        self.injuries = injuries
        self.margin = RidgeMarginModel()
        self.hfa = self.margin.hfa
        self._features: dict[str, dict[str, float]] = {}
        self._teams: list[str] = []
        self._season = 0
        self._fumble_epa = -2.0
        self._obs: dict[str, list[tuple[str, str, int, float, float, int]]] = {}
        self._week_index = 0
        self._st: dict[str, dict[str, list[float]]] = {}
        self.missing_games: list[str] = []

    # --- RatingModel interface -------------------------------------------------

    def start_season(self, season: int, teams: set[str]) -> None:
        self.margin.start_season(season, teams)
        self._season = season
        self._fumble_epa = self.stats.fumble_epa(season)
        self._teams = sorted(teams)
        self._obs = {c: [] for c in PLAY_COMPONENTS}
        self._week_index = 0
        self._st = {t: {c: [] for c in GAME_COMPONENTS} for t in self._teams}
        self._team_qbs = {t: {} for t in self._teams}
        self._qb_prior = {}
        if self.preseason is not None:
            projected = self.preseason.project(season, self._teams)
            self._prior = projected
            self._strength = dict(self.preseason.strengths)
            self.margin.set_prior({t: projected[t]["margin"] for t in self._teams}, self._strength["margin"])
            self._fit()
            return

        last = self._features
        prior = {
            t: {f: self.carryover * last.get(t, {}).get(f, 0.0) for f in FEATURES if f != "margin"}
            for t in self._teams
        }
        for f in FEATURES:
            if f == "margin":
                continue
            mean = sum(prior[t][f] for t in self._teams) / len(self._teams)
            for t in self._teams:
                prior[t][f] -= mean
        self._prior = prior
        self._strength = {}
        for comp in PLAY_COMPONENTS:
            for side in ("off", "def"):
                f = f"{side}_{comp}"
                self._strength[f] = self.prior_plays * self.prior_scale.get(f, 1.0)
        for comp in GAME_COMPONENTS:
            self._strength[comp] = self.st_prior_games
        self._fit()

    def update_week(self, games: list[Game]) -> None:
        if self.qbs is not None:
            self._update_qbs(games)
        self.margin.update_week(games)
        for g in games:
            rows = self.stats.by_game.get(g.game_id)
            if not rows:
                self.missing_games.append(g.game_id)
                continue
            self._absorb(g, rows)
        self._week_index += 1
        self._fit()

    def ratings(self) -> dict[str, float]:
        """Strength with the mix of QBs the team has actually played."""
        return {t: sum(self.weights.get(f, 0.0) * v for f, v in fs.items()) for t, fs in self._features.items()}

    def predict(self, game: Game) -> float:
        adjustments = self.game_adjustments(game)
        return super().predict(game) + sum(self.weights.get(k, 0.0) * v for k, v in adjustments.items())

    def game_adjustments(self, game: Game) -> dict[str, float]:
        """Home minus away for each game-specific adjustment."""
        return {
            "qb": self.qb_diff(game),
            "injuries": self.missing(game.season, game.week, game.home)
            - self.missing(game.season, game.week, game.away),
        }

    def missing(self, season: int, week: int, team: str) -> float:
        return self.injuries.get((season, week, team), 0.0) if self.injuries else 0.0

    def injury_cost(self, season: int, week: int, team: str) -> float:
        """Points this team loses to players ruled out (negative is worse)."""
        return self.weights.get("injuries", 0.0) * self.missing(season, week, team)

    # --- Quarterbacks -------------------------------------------------------------

    def qb_adjustment(self, team: str, qb: str | None) -> float:
        """EPA per play this QB adds over the team's usual QB mix (0 if unknown)."""
        mix = dict(self._team_qbs.get(team, {}))
        if team in self._qb_prior:
            prior_qb, weight = self._qb_prior[team]
            mix[prior_qb] = mix.get(prior_qb, 0.0) + weight
        if self.qbs is None or not qb or not mix:
            return 0.0
        total = sum(mix.values())
        baseline = sum(plays * self.qbs.rating(q) for q, plays in mix.items()) / total
        return self.qbs.rating(qb) - baseline

    def qb_diff(self, game: Game) -> float:
        return self.qb_adjustment(game.home, game.home_qb) - self.qb_adjustment(game.away, game.away_qb)

    def rating_with(self, team: str, qb: str | None) -> float:
        """Team strength in points with this QB starting."""
        return self.ratings()[team] + self.weights.get("qb", 0.0) * self.qb_adjustment(team, qb)

    def primary_qb(self, team: str) -> str | None:
        """The QB with the most plays for this team this season."""
        mix = self._team_qbs.get(team)
        if mix:
            return max(mix, key=mix.get)
        return self._qb_prior.get(team, (None, 0.0))[0]

    def _update_qbs(self, games: list[Game]) -> None:
        # The preseason projection assumed the week-1 starter, so he carries
        # the prior's weight in the team's QB mix.
        for g in games:
            for team, qb in ((g.home, g.home_qb), (g.away, g.away_qb)):
                if qb and team in self._team_qbs and team not in self._qb_prior:
                    self._qb_prior[team] = (qb, self._strength.get("off_pass", self.prior_plays))
        allowed = {t: fs["def_pass"] for t, fs in self._features.items()}
        rows = self.qbs.update_week(games, allowed, self._fumble_epa, self.neutralize_fumbles)
        for r in rows:
            mix = self._team_qbs.get(r["team"])
            if mix is not None:
                mix[r["qb"]] = mix.get(r["qb"], 0.0) + r["plays"]

    # --- Breakdown --------------------------------------------------------------

    def features(self) -> dict[str, dict[str, float]]:
        return {t: dict(fs) for t, fs in self._features.items()}

    def preseason_rating(self, team: str) -> float:
        """The rating this team started the season with, in points."""
        prior = dict(self._prior[team], margin=self.margin._prior[team])
        return sum(self.weights.get(f, 0.0) * prior.get(f, 0.0) for f in FEATURES)

    def breakdown(self, team: str) -> dict[str, float]:
        """A team's rating split into units, in points."""
        fs = self._features[team]
        return {unit: sum(self.weights.get(f, 0.0) * fs[f] for f in feats) for unit, feats in UNITS.items()}

    # --- Internals --------------------------------------------------------------

    def _absorb(self, game: Game, rows: Iterable[dict]) -> None:
        gw = self.garbage_weight
        per_offense: dict[tuple[str, str, str], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
        kick: dict[str, float] = defaultdict(float)
        ret: dict[str, float] = defaultdict(float)
        for r in rows:
            w = gw if r["garbage"] else 1.0
            if r["kind"] in ("pass", "rush"):
                epa = r["epa"]
                if self.neutralize_fumbles:
                    epa += r["fumbles"] * self._fumble_epa - r["fumble_epa"]
                acc = per_offense[(r["offense"], r["defense"], r["kind"])]
                acc[0] += w * epa
                acc[1] += w * r["plays"]
                acc[2] += w * r["interception_epa"]
            elif r["kind"] == "kick":
                kick[r["offense"]] += r["epa"]
            elif r["kind"] == "return":
                ret[r["offense"]] += r["epa"]
                ret[r["defense"]] -= r["epa"]

        for (offense, defense, kind), (epa, plays, int_epa) in per_offense.items():
            if plays <= 0 or offense not in self._st or defense not in self._st:
                continue
            side = 0 if game.neutral else (1 if offense == game.home else -1)
            self._obs[kind].append((offense, defense, side, epa / plays, plays, self._week_index))
            if kind == "pass":
                self._obs["int"].append((offense, defense, side, int_epa / plays, plays, self._week_index))

        for team in (game.home, game.away):
            if team in self._st:
                self._st[team]["st_kick"].append(kick.get(team, 0.0))
                self._st[team]["st_return"].append(ret.get(team, 0.0))

    def _fit(self) -> None:
        teams = self._teams
        n = len(teams)
        index = {t: i for i, t in enumerate(teams)}
        features: dict[str, dict[str, float]] = {t: {} for t in teams}

        for comp in PLAY_COMPONENTS:
            obs = self._obs[comp]
            prior_off = np.array([self._prior[t][f"off_{comp}"] for t in teams])
            prior_def = np.array([self._prior[t][f"def_{comp}"] for t in teams])
            # Unknowns: [mean, home edge, offense x n, defense x n].
            k = 2 + 2 * n
            penalty = np.empty(k)
            penalty[:2] = 1e-6
            penalty[2 : 2 + n] = self._strength[f"off_{comp}"]
            penalty[2 + n :] = self._strength[f"def_{comp}"]
            target = np.concatenate([[0.0, 0.0], prior_off, prior_def])
            a = np.diag(penalty)
            b = penalty * target
            if obs:
                x = np.zeros((len(obs), k))
                y = np.empty(len(obs))
                w = np.empty(len(obs))
                for row, (offense, defense, side, value, plays, week) in enumerate(obs):
                    x[row, 0] = 1.0
                    x[row, 1] = side
                    x[row, 2 + index[offense]] = 1.0
                    x[row, 2 + n + index[defense]] = 1.0
                    y[row] = value
                    w[row] = plays * self._decay(week)
                a = a + x.T @ (x * w[:, None])
                b = b + x.T @ (w * y)
            solution = np.linalg.solve(a, b)
            off = solution[2 : 2 + n] - solution[2 : 2 + n].mean()
            dfn = solution[2 + n :] - solution[2 + n :].mean()
            for t in teams:
                features[t][f"off_{comp}"] = float(off[index[t]])
                features[t][f"def_{comp}"] = float(dfn[index[t]])

        for comp in GAME_COMPONENTS:
            strength = self._strength[comp]
            values = {
                t: (sum(self._st[t][comp]) + strength * self._prior[t][comp])
                / (len(self._st[t][comp]) + strength)
                for t in teams
            }
            mean = sum(values.values()) / n
            for t in teams:
                features[t][comp] = values[t] - mean

        margin = self.margin.ratings()
        for t in teams:
            features[t]["margin"] = margin.get(t, 0.0)
        self._features = features


    def _decay(self, week_index: int) -> float:
        if not self.half_life:
            return 1.0
        return 0.5 ** ((self._week_index - 1 - week_index) / self.half_life)


def fit_weights(model: EpaModel, games: list[Game], seasons: range) -> tuple[dict[str, float], list[tuple]]:
    """Regress game margins on pre-game feature differences over `seasons`.

    Returns the weights and the per-game rows (season, x, y) used, so callers
    can cross-validate without replaying history again.
    """
    from powerrank import backtest

    rows: list[tuple[int, np.ndarray, float]] = []

    def observe(game: Game) -> None:
        if game.season not in seasons:
            return
        fs = model._features
        adjustments = model.game_adjustments(game)
        x = np.array([*(fs[game.home][f] - fs[game.away][f] for f in FEATURES), *(adjustments[k] for k in GAME_ADJUSTMENTS)])
        rows.append((game.season, x, game.home_margin - model.home_field(game)))

    backtest.run(model, games, through_season=seasons.stop - 1, observe=observe)
    return solve_weights(rows), rows


def solve_weights(rows: list[tuple[int, np.ndarray, float]]) -> dict[str, float]:
    """Two-stage fit. Fitting all features at once lets the score margin, which
    overlaps ~90% with pass efficiency, flip the sign of defensive weights. So
    first fit the play-by-play features alone, then one blend between that
    composite and the score margin."""
    x = np.array([r[1] for r in rows])
    y = np.array([r[2] for r in rows])
    blend_names = ("margin", *GAME_ADJUSTMENTS)
    epa_cols = [i for i, f in enumerate(WEIGHT_NAMES) if f not in blend_names]
    blend_cols = [WEIGHT_NAMES.index(f) for f in blend_names]
    stage1, *_ = np.linalg.lstsq(x[:, epa_cols], y, rcond=None)
    blend, *_ = np.linalg.lstsq(np.column_stack([x[:, epa_cols] @ stage1, x[:, blend_cols]]), y, rcond=None)
    weights = dict(zip([WEIGHT_NAMES[i] for i in epa_cols], (stage1 * blend[0]).tolist()))
    weights.update(zip(blend_names, blend[1:].tolist()))
    return {f: weights[f] for f in WEIGHT_NAMES}


def cross_validated_mae(rows: list[tuple[int, np.ndarray, float]]) -> float:
    """Leave-one-season-out error of the margin regression."""
    errors = []
    for season in sorted({r[0] for r in rows}):
        train = [r for r in rows if r[0] != season]
        test = [r for r in rows if r[0] == season]
        beta = np.array([solve_weights(train)[f] for f in WEIGHT_NAMES])
        errors.extend(abs(float(r[1] @ beta) - r[2]) for r in test)
    return float(np.mean(errors))
