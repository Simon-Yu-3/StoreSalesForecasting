# Forecasting store sales with promotion schedules

**Junhui (Simon) Yu · UW ECON 481A · Autumn 2026**

A replenishment planner needs 16 days of forecasts before the next ordering cycle. For Ecuadorian grocer Corporación Favorita, we compare seasonal baselines, independently fitted Prophet models, a pooled gradient-boosted tree, and the pretrained t0-beta time-series model. Within the pooled tree, we test the incremental predictive value of a future promotion schedule after accounting for prior promotions.

**[Programming sample](docs/programming-sample.md)** · [Two-page PDF](output/pdf/StoreSalesForecasting-Programming-Supplement.pdf) · [Executed research report](project.ipynb) · [Reproduce](docs/reproduction.md)

The project's best recorded Kaggle submission scored **0.42172 RMSLE**. A blend of calendar HGB, product-family LightGBM, and t0-beta reduced the first submission's error by **10.7%** on the same 28,512 test rows. In the historical study, the value of the promotion schedule was concentrated in promoted observations, and its effect on shortages and excesses varied by window. [Kaggle results and model recipe](#kaggle-results)

## Run the programming example

The executable sample reads typed CSV data, validates keyed records, joins a true 28-day calendar lag, fits a small HGB model, and evaluates all 16 forecast days from one cutoff. It writes a metrics CSV and forecast figure. The default run uses six generated series, so a reviewer can run the code without competition credentials.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
python examples/programming_sample.py --fixture
python -m pytest tests -q
```

The example produces `results/programming-sample/metrics.csv` and `forecast.png`. It is a compact implementation example. The full ensemble, historical experiments, and scored Kaggle files have separate records below. The installed CLI also runs a generated-data benchmark with `storesales-forecast --fixture --output-dir results/fixture`.

The `competition` extra enables the native LightGBM integration test. GitHub Actions installs that extra and verifies actual HGB and LightGBM fits, keyed forecast export, the installed CLI, and the CSV example. [Programming smoke workflow](.github/workflows/programming-smoke.yml)

[Read the programming sample](docs/programming-sample.md) · [Two-page PDF](output/pdf/StoreSalesForecasting-Programming-Supplement.pdf) · [Executable source](examples/programming_sample.py) · [Full experiment reproduction](docs/reproduction.md)

## Findings

Among the seven configurations in the main promotion study, the development-selected pooled tree with scheduled promotions had the lowest RMSLE in all three evaluation windows. Its pooled evaluation RMSLE was **0.41636**, about **2.2%** lower than the promotion-aware t0-beta model. t0-beta had the lowest daily MAE and WAPE in each evaluation window. A model can therefore rank first on Kaggle's metric while another produces smaller errors in the units planners order.

![RMSLE for the two leading models across the three evaluation windows, alongside the effect of adding past and scheduled promotions to the pooled tree.](files/forecast-evidence.png)

| Configuration | Development RMSLE | Evaluation RMSLE | Evaluation MAE |
| --- | ---: | ---: | ---: |
| HGB · past and scheduled promotions | **0.52069** | **0.41636** | 71.06 |
| HGB · sales and calendar | 0.53746 | 0.42524 | 71.26 |
| t0-beta · past and scheduled promotions | 0.52523 | 0.42565 | **66.59** |
| HGB · past promotions only | 0.53844 | 0.42662 | 71.77 |
| Prophet · past and scheduled promotions | 0.67694 | 0.50242 | 128.24 |
| Seasonal naïve | 0.67618 | 0.56105 | 87.68 |
| Eight-week weekday mean | 0.64981 | 0.59415 | 78.42 |

### How the two leading models compare by date

| Forecast dates | HGB RMSLE | t0-beta RMSLE | HGB daily MAE | t0-beta daily MAE |
| --- | ---: | ---: | ---: | ---: |
| May 1–16, 2017 | **0.45331** | 0.46229 | 86.60 | **81.41** |
| May 17–June 1, 2017 | **0.40928** | 0.42118 | 68.82 | **63.02** |
| June 13–28, 2017 | **0.38350** | 0.39043 | 57.78 | **55.36** |

Both models receive the scheduled promotion counts. HGB's pooled lead on RMSLE also varies within the forecast horizon. HGB leads t0-beta at day 1 (**0.43241** versus **0.45279**), and t0-beta leads at day 16 (**0.42498** versus **0.43489**). Each per-day comparison pools the same 5,346 store–family predictions across the evaluation windows. [Overall scores](files/final-results/scores.csv) · [Lead and group scores](files/final-results/slices.csv)

### Where the promotion schedule helps

With the HGB settings fixed, adding the future schedule to past promotions lowered pooled evaluation RMSLE from **0.42662 to 0.41636**. Most of that gain was on promoted store–family–days. The same rows are used on each side of the comparison.

| Rows in the three evaluation windows | Rows | HGB with past promotions | HGB with past and scheduled promotions |
| --- | ---: | ---: | ---: |
| At least one promotion | 40,647 | 0.35778 | **0.33274** |
| Zero promotions | 44,889 | 0.48052 | **0.47966** |

The schedule is most informative for rows where a promotion is planned. Model rankings also depend on the group. t0-beta has a slight edge on promoted rows (**0.33067** versus HGB's **0.33274**), and HGB has a wider edge on unpromoted rows (**0.47966** versus **0.49621**). The comparison assumes planners would have had the recorded future counts at each cutoff. [Calculated promotion slices](files/final-results/slices.csv) · [Full forecast diagnostics](project.ipynb)

## Explore the data

| Item | Coverage |
| --- | --- |
| Forecast unit | Daily sales at one store for one product family |
| Training data | 3,000,888 observations across 54 stores and 33 families from January 2013 to August 2017 |
| Series and dates | 1,782 store–family series and 1,684 recorded dates. The file omits December 25 in 2013–2016 |
| Sales distribution | Median of 11, mean of 357.78, and 90th percentile of 867 sales units per row |
| Zeros and promotions | 31.30% of recorded rows show zero sales. 20.37% show at least one promotion |
| Kaggle test | 28,512 rows for August 16–31, 2017 |

Two patterns motivate the design. Average panel-wide daily sales were about **63% higher on Sundays than on Thursdays**, so weekday baselines are meaningful comparisons. Promotion coverage also changed over time, from **22.6%** in August 2015 to **36.4%** in August 2016 and **42.3%** in July 2017. A single historical window would miss that variation. The notebook charts both patterns. [Competition data](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data) · [Monthly aggregates](files/data-profile/monthly.csv) · [Weekday aggregates](files/data-profile/weekday.csv)

## How the models were selected

The experiment tunes on six 16-day development windows across 2016–2017, then measures how the selected model transfers to three later 16-day windows. Every model issues the full forecast from the cutoff using up to 512 days of sales history. The pooled model simulates earlier origins whose training labels are already observed. Tests change post-cutoff sales and promotion counts to check that the model sees only the information its variant is allowed to use. [Forecast dates and protocol](files/experiment-plan.md#expanded-evaluation-protocol) · [Run manifest](files/final-results/manifest.json)

The reference HGB settings scored **0.52069** on pooled development RMSLE. A more flexible tree scored **0.52086**, and a more regularized tree scored **0.52324**. The reference won by a small margin, and the experiment holds those settings fixed across the three promotion variants. RMSLE pools the individual squared log errors before taking the square root. MAE counts each absolute unit error directly, while WAPE expresses those errors as a share of recorded sales. [Tuning results](files/final-results/tuning.csv) · [Kaggle metric](https://www.kaggle.com/c/store-sales-time-series-forecasting/overview/evaluation)

## What the forecasts mean for ordering

Aggregating the daily forecasts within each store and product family over 16 days exposes a tradeoff hidden by the overall RMSLE improvement. The table shows the change after adding future promotions to the HGB model that already uses past promotions. A negative value means fewer error units on that side.

| Forecast start | Change in units underforecast | Change in units overforecast |
| --- | ---: | ---: |
| May 1 | −49,553 | +13,723 |
| May 17 | +43,986 | −18,163 |
| June 13 | −35,686 | +8,342 |

Giving an underforecast unit half, equal, or twice the cost of an overforecast unit favored using the schedule in **two of three** windows under each assumption. For an operational test, planners could run both forecasts alongside current orders and record the plan available at each cutoff, stockouts, inventory on hand, and waste. That would show whether smaller forecast errors improve actual ordering outcomes. [Cost sensitivity and Decision Loop](project.ipynb)

## Holiday and payday follow-up

The competition records stores' cities and states, the dates when holidays were actually observed, and a regular public-sector payday schedule. We added holiday indicators, a payday indicator, or both to the same HGB model, holding the 16-day forecast dates, promotion inputs, and model settings fixed. For example, the national holiday nominally on May 24, 2017 is flagged on its observed date, May 26. [Calendar features and assumptions](files/experiment-plan.md#calendar-follow-up) · [Kaggle data documentation](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data)

| Features added to the reference | Development RMSLE | Evaluation RMSLE |
| --- | ---: | ---: |
| Reference with weekday and month fields | **0.52069** | 0.41636 |
| Local, regional, and national holidays | 0.52137 | 0.41038 |
| Payday on the 15th and at month-end | 0.52199 | 0.41506 |
| Holidays and payday | 0.52162 | **0.40885** |

The combined features improved evaluation RMSLE by **1.8%**, with better results in **two of three** evaluation windows. The holiday-affected rows were **6.5%** of the evaluation panel and accounted for about **64%** of the net reduction in squared log error. Every augmented variant scored worse on pooled development RMSLE. The study retains its original HGB selection, and the calendar variant is revisited in the competition follow-up below. [Calendar scores](files/calendar-results/scores.csv) · [Holiday and payday slices](files/calendar-results/slices.csv)

## Kaggle results

| Submitted | Forecast model | Official test RMSLE |
| --- | --- | ---: |
| November 16, 2026 | HGB with past and scheduled promotions | 0.47212 |
| November 17, 2026 | 75% calendar HGB + 25% promotion-aware t0-beta | 0.43811 |
| November 19, 2026 | 50% calendar HGB700 + 25% family LightGBM + 25% t0-beta | **0.42172** |

All three submissions predict August 16–31, 2017. Both ensembles average `log1p` sales predictions and use `expm1` to recover sales units. The HGB component adds effective holidays and public-sector paydays to the features used for the first submission. The t0-beta component uses the same pretrained checkpoint and promotion inputs as the main study. [Kaggle submission history](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/submissions) · [V1 and V2 records](files/competition-results/manifest.json) · [V3 score and checksum](files/refinement-results/submission-manifest.json)

For V2, weights were compared on three May–June windows using pooled RMSLE. The selected recipe scored **0.40462** versus **0.41636** for the first model. With its weights fixed, the blend improved all three later historical windows, reducing pooled RMSLE from **0.39644 to 0.38903**. These dates had already been used in the earlier benchmark, so the later-window comparison is a retrospective check. Kaggle scores the submissions on the August 16–31 test period. [Weight grid](files/competition-results/blend-weight-scores.csv) · [Six historical comparisons](files/competition-results/backtest.csv)

On those later historical windows, pooled WAPE fell from **13.55% to 12.46%**. That measure sums the absolute sales-unit errors and divides by total recorded sales. An inventory pilot would still need to measure stockouts, waste, and ordering costs. [Forecast errors and sales totals by window](files/competition-results/backtest.csv)

### Third Kaggle submission

The third submission combines **50% calendar HGB with 700 boosting iterations, 25% product-family LightGBM, and 25% promotion-aware t0-beta** on the log-sales scale. LightGBM fits a separate model for each of the 33 families. Its features add same-weekday summaries, recent sales and promotion changes, store location and type, and prior-year sales. All sales features stop at the forecast origin. Scheduled promotions cover the 16-day forecast horizon.

Four LightGBM configurations were screened on a May–June selection window. The family model contributed the most to the fixed three-model blend, while a global model with a longer, recency-weighted history had the best standalone score. The subsequent weight grid used all three May–June windows and selected the weights above, reducing pooled RMSLE from **0.40462 to 0.40067**. [Initial model screening](files/refinement-results/screening.csv) · [Blend selection](files/refinement-results/selection-grid.csv)

With the recipe fixed, pooled RMSLE on the three later historical windows fell from **0.38903 to 0.38339**, a **1.45%** reduction. Pooled WAPE fell from **12.46% to 11.21%**. Both measures improved in each of the three later windows. These dates had already appeared in the project, so the comparison is retrospective. [Six-window comparison](files/refinement-results/backtest.csv)

| Forecast dates | V2 RMSLE | V3 RMSLE |
| --- | ---: | ---: |
| June 29–July 14, 2017 | 0.37125 | **0.36700** |
| July 15–30, 2017 | 0.38914 | **0.38726** |
| July 31–August 15, 2017 | 0.40593 | **0.39534** |

Kaggle scored V3 at **0.42172 RMSLE** on November 19, 2026, **3.74%** below V2 and **10.68%** below V1. The completed submission and score were recorded from the screenshot supplied by the student. The frozen historical selection archive documents the recipe before submission, and the separate submission record holds the Kaggle result. [Historical model recipe](files/refinement-results/manifest.json) · [Submitted CSV checksum and score evidence](files/refinement-results/submission-manifest.json)

## Earlier experiments

The archived PDF and dashboard summarize the preliminary June–August 2017 experiments. The final report above includes the expanded evaluation and three Kaggle submissions. [Original benchmark](files/official-results/scores.csv) · [Pooled model follow-up](files/global-model-comparison.md) · [Archived dashboard](files/forecast-dashboard.html) · [Archived preliminary report (PDF)](files/store-sales-research-report.pdf)

The repository follows the [ECON 481A student project template](https://github.com/eisenhauerIO/projects-student-template).
