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

## Weekly update

`.github/workflows/weekly-rankings.yml` runs every Tuesday after Monday Night
Football, with retries Tuesday evening and Wednesday in case nflverse hasn't
published the play-by-play yet. It runs the tests, regenerates
`output/rankings.json`, and commits it if anything changed. It can also be run
by hand from the Actions tab.

## Models

| model   | what it is |
|---------|------------|
| `epa`   | **Default.** Opponent-adjusted play-by-play efficiency (pass, rush, interceptions, special teams) blended with score margin. Gives each team an offense / defense / special teams / scoring breakdown. |
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
| epa | 10.82 | 10.17 |
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
- The remaining ~0.4 point gap to the closing line is mostly information the
  market has and this model doesn't yet: injuries, quarterback changes, and
  offseason moves (plan steps 3 and 4).
