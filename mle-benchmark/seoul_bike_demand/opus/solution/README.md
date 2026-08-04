# t5_bike — Seoul hourly public-bike demand (RMSE)

## Reproduce
```bash
python solution/train_predict.py     # writes outputs/submission.csv
```
Only `train.csv` / `test.csv` are used. Deps: pandas, numpy, lightgbm (scikit-learn only in
the exploratory `exp05` script).

## Files
| file | purpose |
|---|---|
| `common.py` | data loading + feature engineering (single source of truth) |
| `train_predict.py` | **final pipeline**: trains the chosen ensemble, writes the submission |
| `exp01…exp08` | experiment log (each is runnable and prints the CV tables cited below) |

## Validation protocol
Rolling-origin (expanding-window) chronological CV: 8 folds, validation start every 21 days
from 2018-03-15, 28-day validation window, training strictly on earlier timestamps. This
mirrors the task (test = the period immediately after the training range) and, crucially,
reproduces the "validation month / day-of-year never seen in training" situation.
Reported numbers are pooled RMSE over all folds.

## Key findings
1. **`functioning_day == 'No'` ⇒ target is exactly 0** (72/72 train rows). Those rows are
   dropped from training and their predictions hard-set to 0. 223 of 1752 test rows (12.7 %).
2. **Calendar-seasonal features are harmful.** The test span (Sep 19 – Nov 30, doy 262–334)
   lies completely outside the training day-of-year range, so trees can only clamp to the
   nearest seen bucket (December). Adding `month/day/doy/doy_sin/doy_cos` cost 286→304 pooled
   RMSE and blew one fold up to 505 vs 340 (`exp02`, `exp05`). Seasonality is carried by the
   weather channels instead. All test weather values lie inside the training ranges, so the
   model interpolates rather than extrapolates.
3. **No forced trend extrapolation.** At matched temperature the level does grow through the
   year (weekday daytime, 15–20 °C: Mar–Apr 1172 → Sep 1757), but leave-one-month-out
   diagnostics show a weather-only model is already well calibrated (level ratio 0.97–1.12 for
   Mar–Sep), and an explicit `log-level − β·t` trend offset helped only the earliest
   long-horizon fold while badly hurting the later ones (`exp03`, `exp04`). Recency weighting
   also did not help. No post-hoc scaling factor was applied since none could be validated.
4. **Objective matters more than hyper-parameters.** Log-link count objectives model the
   conditional mean, which is what RMSE wants; L2 on `log1p(y)` targets the geometric mean
   and under-predicts. Pooled RMSE (`exp06`): poisson 275.7, tweedie(1.3) 278.6,
   sqrt-L2 281.3, log-L2 286.6, raw-L2 312.0.
5. **Final model** — equal-weight blend, each member averaged over 5 seeds:
   * Poisson, `num_leaves=31, min_data_in_leaf=30, feature_fraction=0.5`, 1800 rounds
   * Poisson, `num_leaves=15, min_data_in_leaf=50`, 2500 rounds
   * Tweedie(p=1.3), `num_leaves=31, min_data_in_leaf=30`, 1400 rounds
   Pooled CV RMSE **266.5** (worst fold 326) vs 286.6 for the first log-L2 baseline.
   Held-out last 28 days of train: RMSE **190.2**.

## Features (`common.py`)
Raw weather, `hour`, `dayofweek`, weekend/holiday/non-working flags; rain & snow flags,
`log1p(rain)`, temp×humidity, temp−dew-point, an apparent-temperature proxy; ±1 h and −3 h
lags/leads of temperature, rainfall, humidity, solar radiation; 3/6/24 h rolling rainfall and
24 h rolling temperature; daily aggregates (temp mean/max/min, rain sum, solar sum, humidity
mean) and the deviation of the hour from its daily mean temperature.
Lags/leads and daily aggregates use **weather only** — never the target — so there is no
label leakage; test-period weather is given by the task, exactly as in the source dataset.

## Sanity checks on the submission
1752 rows, ids identical to `sample_submission.csv` (order preserved), no NaN, all
non-negative, 0 for every non-functioning hour. Monthly mean of the predictions on
functioning hours: Sep 1010, Oct 675, Nov 430, with the expected 8 h / 18 h commute peaks.
