# power-rankings

NFL power rankings built to predict, not to reward records. Each team's rating
is the number of points it would beat a league-average team by on a neutral
field. See [PLAN.md](PLAN.md) for the full design.

## Setup

```sh
pip install -r requirements.txt
```

Data comes from [nflverse](https://github.com/nflverse) and is cached in
`data/cache/` for six hours. Pass `--refresh` to re-download.

## Usage

```sh
python -m powerrank rank                 # current ratings + next week's lines, writes output/rankings.json
python -m powerrank rank --model ridge   # same, using a baseline
python -m powerrank backtest             # score every model week by week vs. the closing line
python -m powerrank fit                  # refit the epa model's weights (once per offseason)
python -m pytest                         # tests
```

`rank` only counts fully completed weeks, so running it mid-week never mixes
in Thursday's game.

## Rankings output

`output/rankings.json` lists each team's `rating` (strength for its next game,
with the projected starting QB and that week's injury report, which is what
the rank uses), `healthy_rating` (with its main QB and nobody ruled out), a
`breakdown` into offense / defense / special teams / scoring / quarterback /
injuries that sums to `rating`, its `preseason_rating`, offseason changes,
record, and next week's model and market lines.

## Weekly update

`.github/workflows/weekly-rankings.yml` runs every Tuesday after Monday Night
Football, with retries Tuesday evening and Wednesday in case nflverse hasn't
published the play-by-play yet. It runs the tests, regenerates
`output/rankings.json`, and commits it if anything changed. It can also be run
by hand from the Actions tab.

## Models

| model   | what it is |
|---------|------------|
| `epa`   | **Default.** Opponent-adjusted play-by-play efficiency (pass, rush, interceptions, special teams) blended with score margin, starting each season from a preseason projection, and adjusted for who starts at QB and who is ruled out. Gives each team an offense / defense / special teams / scoring / QB / injuries breakdown. |
| `ridge` | Baseline: opponent-adjusted point differential, with last season as a prior |
| `elo`   | Baseline: FiveThirtyEight-style Elo, in points |

## Results

Mean absolute error of the predicted margin, predicting each week using only
earlier results. Tuned on 2006-2015, tested on 2016-2025, which the model
never saw while being tuned.

| | 2006-2015 (tuning) | 2016-2025 (held out) |
|---|---|---|
| elo | 10.89 | 10.19 |
| ridge | 10.83 | 10.17 |
| epa | 10.77 | 10.08 |
| closing line | 10.71 | 9.78 |

What the tuning found:
- Play-by-play efficiency predicts better than score margin in weeks 5-9 but
  worse early in the season, and the blend comes out even with score margin
  overall. Its main value so far is the unit breakdown and the foundation for
  quarterback adjustments.
- Down-weighting garbage time, neutralizing fumble recoveries, separate
  shrinkage for defense and rushing, and weighting recent games more all moved
  the error by less than 0.01 points, below the noise. Recency weighting made
  it worse.
- Preseason projections (last two seasons, QB change and whether the new QB
  is a veteran, coach change, draft capital) beat simple carryover for teams
  that changed quarterback (5.26 vs 5.44 RMSE on full-season strength,
  2016-2025) and tie it overall. A new QB cuts the carryover of last
  season's passing offense by more than half; a new coach does the same for
  the run game. Prior strengths come from empirical Bayes (play-to-play noise
  divided by projection error), and cross-validation confirms they are
  right: halving or doubling them is worse. Overall game accuracy is
  unchanged: public offseason information explains only ~20% of how good a
  team turns out to be.
- **Quarterbacks are the biggest single improvement** (10.18 -> 10.10
  held out). Each QB has a career rating (EPA per play above what his
  opponents usually allow, older plays fading with a 40-week half-life,
  shrunk toward a below-average unknown). A game's adjustment is the
  starter's rating minus the mix of QBs behind the team's numbers. In the 329
  held-out games where this moved the line 2+ points, error fell from 10.54
  to 10.06 and the model's lines got much closer to the market's.
- Non-QB injuries (recent snap share of players ruled Out/Doubtful) help a
  little: ~0.8 points per full-time starter out, 0.01 points of held-out
  error. Separate weights by position overfit the few seasons of data.
  Injury reports come out Wednesday-Friday, so the Tuesday rankings
  usually can't include the coming week's; they do include projected
  starting QBs.
- Preseason market win totals would be the biggest single improvement for
  the first month, but are left out on purpose so the model stays
  independent of Vegas.
- The remaining ~0.4 point gap to the closing line is mostly information the
  market has and this model doesn't yet: injuries, quarterback changes, and
  offseason moves (plan steps 3 and 4).
