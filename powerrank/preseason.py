"""Preseason projections: each team's starting point before any games.

For every model feature, a regression projects this season's full-season value
from information available before week 1:
  - the team's value last season and two seasons ago
  - whether the week-1 starting QB differs from last season's main starter,
    and whether the new one is a veteran (16+ career starts)
  - whether the head coach changed
  - draft capital from this year's draft
The score-margin projection also gets last season's efficiency, which predicts
next season's margin better than margin itself does.

How hard a projection pulls on in-season estimates is empirical Bayes:
strength = (noise of one play or game) / (variance of the projection's error),
so accurate projections hold longer and noisy units update faster.

Market win totals are deliberately not used: the model stays independent of
Vegas so its disagreement with the market means something.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import polars as pl

from powerrank import backtest
from powerrank.data import CACHE_DIR, TEAM_ALIASES, Game
from powerrank.models.epa import FEATURES, GAME_COMPONENTS, PLAY_COMPONENTS, EpaModel, TeamGameStats

VETERAN_STARTS = 16
# Draft picks from Pro Football Reference use their own abbreviations.
DRAFT_ALIASES = {
    "GNB": "GB", "KAN": "KC", "LAR": "LA", "LVR": "LV", "NOR": "NO", "NWE": "NE",
    "SDG": "LAC", "SFO": "SF", "TAM": "TB", **TEAM_ALIASES,
}
EPA_FOR_MARGIN = ("off_pass", "def_pass", "off_rush", "def_rush")
OFFSEASON = ("qb_changed", "veteran_qb", "coach_changed", "draft")


def pick_value(pick: int) -> float:
    """Top-heavy value curve: #1 ~1.0, #32 ~0.6, #100 ~0.2, #200 ~0.04."""
    return math.exp(-pick / 64)


def draft_values(refresh: bool = False) -> dict[tuple[int, str], float]:
    path = CACHE_DIR / "draft_picks.parquet"
    if path.exists() and not refresh:
        df = pl.read_parquet(path)
    else:
        import nflreadpy

        df = nflreadpy.load_draft_picks().select("season", "pick", "team")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.write_parquet(path)
    totals: dict[tuple[int, str], float] = defaultdict(float)
    for r in df.iter_rows(named=True):
        totals[(r["season"], DRAFT_ALIASES.get(r["team"], r["team"]))] += pick_value(r["pick"])
    return dict(totals)


def offseason_info(games: list[Game], drafts: dict[tuple[int, str], float]) -> dict[tuple[int, str], dict[str, float]]:
    """QB, coach and draft changes for each (season, team), known before its first game."""
    starters: dict[tuple[int, str], list[str]] = defaultdict(list)
    coaches: dict[tuple[int, str], list[str]] = defaultdict(list)
    career_starts: Counter[str] = Counter()
    starts_before: dict[tuple[int, str], int] = {}
    for season in sorted({g.season for g in games}):
        season_games = [g for g in games if g.season == season]
        for g in season_games:
            for team, qb, coach in ((g.home, g.home_qb, g.home_coach), (g.away, g.away_qb, g.away_coach)):
                if qb and (season, qb) not in starts_before:
                    starts_before[(season, qb)] = career_starts[qb]
                starters[(season, team)].append(qb)
                coaches[(season, team)].append(coach)
        for g in season_games:
            if g.played:
                career_starts.update(q for q in (g.home_qb, g.away_qb) if q)

    info = {}
    for (season, team), qbs in starters.items():
        last_qbs = starters.get((season - 1, team))
        last_coaches = coaches.get((season - 1, team))
        week1_qb = qbs[0]
        if last_qbs:
            main_qb = Counter(q for q in last_qbs if q).most_common(1)
            qb_changed = float(bool(main_qb) and week1_qb != main_qb[0][0])
            coach_changed = float(coaches[(season, team)][0] != last_coaches[-1])
        else:
            qb_changed = coach_changed = 0.0
        veteran = float(qb_changed and starts_before.get((season, week1_qb), 0) >= VETERAN_STARTS)
        info[(season, team)] = {
            "qb_changed": qb_changed,
            "veteran_qb": veteran,
            "coach_changed": coach_changed,
            "draft": drafts.get((season, team), 0.0),
        }
    # Draft capital relative to the league that year.
    for season in {s for s, _ in info}:
        keys = [k for k in info if k[0] == season]
        mean = sum(info[k]["draft"] for k in keys) / len(keys)
        for k in keys:
            info[k]["draft"] -= mean
    return info


@dataclass
class SeasonEstimates:
    """Full-season feature values fitted with (almost) no prior, plus noise levels."""

    values: dict[int, dict[str, dict[str, float]]]
    # Variance of one play (play components) or one game (others) around the fit.
    noise: dict[str, float]
    # Typical sampling variance of a full-season estimate, per feature.
    sampling: dict[str, float]


def season_estimates(stats: TeamGameStats, games: list[Game]) -> SeasonEstimates:
    model = EpaModel(stats=stats, weights={}, prior_plays=1e-3, carryover=0.0, st_prior_games=1e-3, prior_scale={})
    model.margin.prior_games = 1e-3
    values: dict[int, dict[str, dict[str, float]]] = {}
    noise_sums: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    sampling: dict[str, list[float]] = defaultdict(list)

    played = [g for g in games if g.played]
    for season in sorted({g.season for g in played}):
        season_games = [g for g in played if g.season == season]
        model.start_season(season, {t for g in season_games for t in (g.home, g.away)})
        for _season, _week, week_games in backtest.group_weeks(season_games):
            model.update_week(week_games)
        features = model.features()
        values[season] = features
        _accumulate_noise(model, features, season_games, noise_sums, sampling)

    noise = {f: s / n for f, (s, n) in noise_sums.items()}
    return SeasonEstimates(
        values=values,
        noise=noise,
        sampling={f: noise[f] * float(np.mean(v)) for f, v in sampling.items()},
    )


def _accumulate_noise(model: EpaModel, features, season_games, noise_sums, sampling) -> None:
    for comp in PLAY_COMPONENTS:
        obs = model._obs[comp]
        if not obs:
            continue
        resid = np.array([v - features[o][f"off_{comp}"] - features[d][f"def_{comp}"] for o, d, _, v, _, _ in obs])
        side = np.array([s for _, _, s, _, _, _ in obs], dtype=float)
        plays = np.array([p for _, _, _, _, p, _ in obs])
        design = np.column_stack([np.ones_like(side), side]) * np.sqrt(plays)[:, None]
        coef, *_ = np.linalg.lstsq(design, resid * np.sqrt(plays), rcond=None)
        resid = resid - coef[0] - coef[1] * side
        for f in (f"off_{comp}", f"def_{comp}"):
            noise_sums[f][0] += float(np.sum(plays * resid**2))
            noise_sums[f][1] += len(obs)
            per_team = defaultdict(float)
            for (o, d, _, _, p, _) in obs:
                per_team[o if f.startswith("off") else d] += p
            sampling[f].extend(1.0 / p for p in per_team.values())
    for comp in GAME_COMPONENTS:
        for team, per_comp in model._st.items():
            vals = per_comp[comp]
            if len(vals) > 1:
                noise_sums[comp][0] += float(np.var(vals, ddof=1)) * len(vals)
                noise_sums[comp][1] += len(vals)
                sampling[comp].append(1.0 / len(vals))
    margin = model.margin
    games_per_team: Counter[str] = Counter()
    for g in season_games:
        expected = features[g.home]["margin"] - features[g.away]["margin"] + margin.home_field(g)
        capped = max(-margin.margin_cap, min(margin.margin_cap, g.home_margin))
        noise_sums["margin"][0] += (capped - expected) ** 2
        noise_sums["margin"][1] += 1
        games_per_team.update((g.home, g.away))
    sampling["margin"].extend(1.0 / n for n in games_per_team.values())


class Preseason:
    """Projects each team's features before week 1 and says how hard to hold them."""

    def __init__(
        self,
        coefficients: dict[str, list[float]],
        strengths: dict[str, float],
        estimates: SeasonEstimates,
        offseason: dict[tuple[int, str], dict[str, float]],
    ):
        self.coefficients = coefficients
        self.strengths = strengths
        self.estimates = estimates
        self.offseason = offseason

    @staticmethod
    def predictors(feature: str, season: int, team: str, estimates: SeasonEstimates, offseason) -> list[float]:
        last = estimates.values.get(season - 1, {}).get(team, {})
        last2 = estimates.values.get(season - 2, {}).get(team, {})
        info = offseason.get((season, team), dict.fromkeys(OFFSEASON, 0.0))
        prev = last.get(feature, 0.0)
        row = [
            prev,
            last2.get(feature, 0.0),
            info["qb_changed"] * prev,
            info["coach_changed"] * prev,
            *(info[k] for k in OFFSEASON),
        ]
        if feature == "margin":
            row += [last.get(f, 0.0) for f in EPA_FOR_MARGIN]
        return row

    @classmethod
    def fit(
        cls,
        estimates: SeasonEstimates,
        offseason: dict[tuple[int, str], dict[str, float]],
        seasons: range,
    ) -> Preseason:
        coefficients, strengths = {}, {}
        for f in FEATURES:
            x, y = [], []
            for season in seasons:
                for team, values in estimates.values.get(season, {}).items():
                    x.append(cls.predictors(f, season, team, estimates, offseason))
                    y.append(values[f])
            x_arr, y_arr = np.array(x), np.array(y)
            beta, *_ = np.linalg.lstsq(x_arr, y_arr, rcond=None)
            coefficients[f] = beta.tolist()
            # Error of the projection, minus the part that is just noise in the
            # full-season target, is the real uncertainty of the prior.
            error_var = float(np.var(y_arr - x_arr @ beta))
            prior_var = max(error_var - estimates.sampling[f], 0.1 * error_var)
            strengths[f] = estimates.noise[f] / prior_var
        return cls(coefficients, strengths, estimates, offseason)

    def project(self, season: int, teams: list[str]) -> dict[str, dict[str, float]]:
        projected = {
            t: {
                f: float(np.dot(self.coefficients[f], self.predictors(f, season, t, self.estimates, self.offseason)))
                for f in FEATURES
            }
            for t in teams
        }
        for f in FEATURES:
            mean = sum(projected[t][f] for t in teams) / len(teams)
            for t in teams:
                projected[t][f] -= mean
        return projected

    def to_json(self) -> dict:
        return {
            "coefficients": {f: [round(c, 4) for c in cs] for f, cs in self.coefficients.items()},
            "strengths": {f: round(s, 2) for f, s in self.strengths.items()},
        }


def build(stats: TeamGameStats, games: list[Game], seasons: range | None = None, saved: dict | None = None) -> Preseason:
    """Fit on `seasons`, or rebuild from saved coefficients."""
    estimates = season_estimates(stats, games)
    offseason = offseason_info(games, draft_values())
    if saved is not None:
        return Preseason(saved["coefficients"], saved["strengths"], estimates, offseason)
    return Preseason.fit(estimates, offseason, seasons)
