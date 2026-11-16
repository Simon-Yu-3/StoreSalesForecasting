# Can a pooled tree model close the forecasting gap?

The first benchmark compared t0-beta with Prophet and seasonal baselines. Prophet fits each store–family separately. This follow-up asks whether a conventional model that learns jointly from all stores and product families narrows the gap.

## Result

| Configuration | Development 1 | Development 2 | Pooled development | Final holdout | Holdout MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prophet + promotions | 0.46318 | 0.46576 | 0.46448 | 0.48376 | 108.53 |
| t0-beta + promotions | 0.38570 | 0.39344 | 0.38959 | **0.41312** | **71.45** |
| Pooled gradient boosting | 0.38651 | 0.40457 | 0.39564 | 0.49982 | 79.43 |
| Pooled gradient boosting + promotions | **0.37457** | 0.39446 | **0.38464** | 0.41904 | 76.28 |

The score is RMSLE except in the last column, which is mean absolute error in daily sales units. Development RMSLE pools the squared log errors across both windows before taking a square root. Each window contains the same 28,512 rows per model. [Original scores](official-results/scores.csv) · [Follow-up scores](global-comparison/scores.csv)

**The ranking changes over time.** Pooled boosting with promotions has the lowest development RMSLE. On July 31–August 15, t0-beta with promotions has the lowest RMSLE and MAE among these four configurations. Its holdout RMSLE is 14.6% lower than Prophet with promotions and 1.4% lower than pooled boosting with promotions. The latter is the stronger conventional reference for this comparison.

## Forecasting design

Fit a histogram gradient boosting regressor to log-transformed sales, with store and family categories, the forecast lead, calendar features, same-weekday sales lags, and trailing sales summaries. The promotion-aware variant adds the target day's scheduled promotions and the trailing promotion mean. Its predictions assume the target-date counts were available at each cutoff. The two variants share the remaining features, the learning rate, and the fixed 150 boosting iterations. [Model documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html)

At each cutoff, the preceding 512 calendar days define the training window. The model learns from simulated 16-day forecasts beginning 15 days apart. Each training feature uses sales observed by its simulated origin, and all training target dates fall at or before the relevant cutoff. Each fit has 739,530 labeled training rows pooled across the panel. The leakage regression test changes post-cutoff sales and checks that features and training labels are identical. [Implementation](../auxiliary/global_benchmark.py) · [Run manifest](global-comparison/manifest.json)

The model was added after the original holdout results were reviewed. Its comparison on the same dates is retrospective. Each fit uses only the data available at its cutoff. A separate evaluation period or an official Kaggle result would add evidence about generalization beyond these windows.

## Replenishment implications

For each of 1,782 store–family series, sum the 16 daily forecasts and compare that total with the 16 observed days. Sum shortfalls and excesses separately across the series so that one store's excess cannot cancel another store's shortfall. The following table uses the final historical window.

| Pooled tree variant | Total underforecast units | Total overforecast units | Daily WAPE |
| --- | ---: | ---: | ---: |
| History only | 484,057 | 895,517 | 17.00% |
| With promotion counts | 384,911 | 907,169 | 16.33% |

The schedule-aware model produces about **99,146 fewer underforecast units** and **11,652 more overforecast units** at the 16-day series level. WAPE is total daily absolute error divided by total observed sales. This directional comparison concerns forecast errors on recorded sales. Actual stockouts, waste, and costs require inventory, replenishment, and demand data.

## Reproduce

With the course environment active, run the command from the repository root. The benchmark saves the exact configuration and file hashes.

```bash
OMP_NUM_THREADS=4 python -m auxiliary.global_benchmark \
  --data data/raw/store-sales-time-series-forecasting.zip \
  --output-dir results/global-comparison
```

To prepare a separate Kaggle test forecast from the configuration favored by the pooled development windows, rerun the command with `--prediction-model global_hgb_promotion` and a new output directory. The resulting `submission.csv` contains forecasts for the official test IDs. The project reports local backtests.
