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
python -m powerrank rank --model elo     # same, using the Elo baseline
python -m powerrank backtest             # score every model week by week vs. the closing line
python -m pytest                         # tests
```

## Models

| model   | what it is |
|---------|------------|
| `elo`   | FiveThirtyEight-style Elo, in points |
| `ridge` | Opponent-adjusted point differential, regression with last season as a prior |

These are the baselines. Every later component has to beat them on seasons it
was not tuned on.

## Current baseline results

Mean absolute error of the predicted margin, predicting each week using only
earlier results. Tuned on 2006-2015, tested on 2016-2025.

| | 2006-2015 (tuning) | 2016-2025 (held out) |
|---|---|---|
| elo | 10.89 | 10.19 |
| ridge | 10.83 | 10.17 |
| closing line | 10.71 | 9.78 |
