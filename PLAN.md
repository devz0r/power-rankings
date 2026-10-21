# NFL Power Rankings: Plan

## Decisions
- **Language:** Python (nflreadpy for nflverse data)
- **Markets:** model stays independent of Vegas; a column shows where it disagrees with the market
- **Output:** static website with two views: a ranked list and a tier list (S through F, dark theme)

## Core idea
The ranking is predictive: who would win on a neutral field next Sunday? Each team gets one number, the points it would beat a league-average team by on a neutral field right now (e.g. +7.5). Rank is that number sorted. Because it is in points, it can be tested against future scores and Vegas spreads. Résumé measures are shown in separate columns and do not affect the order.

## What it considers

**1. Play-by-play efficiency (main input)**
- EPA per play and success rate, split into offense, defense, special teams, and into passing and running.
- Opponent-adjusted with a regression that solves all 32 teams together.
- Garbage time down-weighted using win probability.
- Passing weighted above running, offense above defense; backtest sets exact weights.

**2. Removing luck**
- Fumble recoveries counted at the expected recovery rate.
- Interceptions pulled toward the QB's long-term rate.
- One-score record, red-zone and third-down rates, opponent field-goal accuracy pulled toward league average.
- Expected wins vs. actual wins from points scored and allowed.

**3. Scores**
- Opponent-adjusted point differential with blowouts capped. Secondary input.

**4. Preseason starting point**
- Last season's rating pulled toward average, adjusted for QB, coach and coordinator changes and roster continuity.
- Preseason win totals as a prior.
- Weight fades as games are played; the backtest decides how fast.

**5. Who is actually playing**
- QB-specific rating that changes when the starter changes.
- Injury cost by position from injury reports and snap counts.
- Two numbers per team: healthy strength and this week's strength. Rank uses this week's.
- Trades and signings move value between teams.

**6. Recent games**
- Slightly more weight on recent games, tuned by backtest. No simple "last three games" momentum.

**7. Betting markets**
- Closing spreads are the benchmark the model is scored against, not an input.
- Output includes a **Vegas disagreement** column: model rating vs. the market-implied rating, per team.

**8. Game context (for game predictions only, not the rank)**
- Home field (about 1.5 to 2 points), rest and byes, travel, weather.

**Résumé columns (shown, not ranked on)**
- Record, adjusted point differential, strength of record, best wins and worst losses, remaining schedule difficulty.
- Playoff, division, bye and Super Bowl odds from season simulation.

**Left out on purpose**
- Head-to-head results do not override the rank. The game is already in the data at its proper weight, and the output will say so.
- Storylines, "knows how to win", and reputation.

## Uncertainty and tiers
- Each rating has a range. Show things like the chance a team is truly top 5.
- Tier lines (S, A, B, C, D, E, F) are drawn where gaps between teams exceed the noise, not as equal-sized groups of the 32 teams.

## Explaining moves
- Weekly breakdown per team of why it moved, with a short write-up.

## Proving it works
- **Data:** nflverse play-by-play, schedules with closing lines, injuries, snap counts, rosters.
- **Replay:** rerun past seasons week by week using only information available at the time.
- **Scoring:** error on next week's margins, win-probability accuracy, comparison with closing lines. Tune on older seasons, test on recent unseen ones.
- **Baseline:** simple Elo or adjusted point differential. Anything that does not beat it on unseen seasons is cut.

## Website
- Static site reading one JSON file of ratings produced by the Python pipeline.
- **Ranked view:** table with rank, team, rating, range, résumé columns, Vegas disagreement.
- **Tier view:** seven rows S to F, dark theme, team logos or names placed by tier.
- A toggle switches between the two views.

## Build order
1. Data pipeline, backtest harness, simple baseline. **Done:** Elo and ridge baselines; see README for results.
2. Opponent-adjusted EPA model with garbage-time and luck adjustments. **Done:** ties the score baseline on held-out seasons; adjustments tested and measured as noise-level. See README.
3. Preseason priors and weekly updating. **Done:** projections from the last two seasons, QB/coach changes and draft capital, with empirical-Bayes prior strengths. Helps QB-change teams, ties overall. Win totals left out to stay independent of Vegas.
4. QB and injury adjustments. **Done:** career QB ratings swap in the actual starter (biggest gain so far, held-out MAE 10.18 -> 10.08); non-QB injuries add a little; healthy vs this-week ratings. Injury reports aren't out by Tuesday.
5. Résumé columns, uncertainty, season simulation, Vegas disagreement. **Done:** strength of record, best win / worst loss, remaining schedule, calibrated 80% ranges and chance of being #1/top 5, natural-break tiers, 10,000-season playoff simulation, market-implied ratings, rank changes. Saturday injury-report run added.
6. Export to JSON and build the static site (ranked and tier views).
7. Weekly run after Monday Night Football. **Done:** GitHub Actions, Tuesdays with retries.

## Model and effort
- Modeling and backtest: Opus 5.5, high effort.
- Website and tier list: Sonnet 5.5, medium effort.
