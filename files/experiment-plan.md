# Experiment plan

## Question and decision

Which methods produce the most accurate 16-day-ahead daily sales forecasts for the typical store–product-family observation at Corporación Favorita? How does the ranking change when planners can supply scheduled promotions? The decision is which forecasting process to use as an input to short-run replenishment or staffing. Forecast accuracy under Kaggle's loss does not measure stockouts, profit, or the causal effect of a promotion. The project follows the course's [Decision Loop](https://eisenhauerio.github.io/courses-business-decisions/decision-loop/index.html) by connecting a measurable forecast objective to a decision and a repeatable evaluation.

On October 23, 2026, Professor Philipp Eisenhauer approved the Store Sales topic and asked for a comparison of conventional methods and LLM-based time-series foundation models. He suggested Prophet and The Forecasting Company as starting points. This paraphrase records the approval without republishing the private correspondence.

## Evaluation rules and preliminary benchmark

- The observation is daily unit sales for one store–family pair. Sales can be fractional. Run all methods on the same keyed forecast rows and same date cutoffs. Read and check the horizon against the official test dates before a competition fit. [Competition data](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data)
- The preliminary benchmark used up to 512 calendar days of history at each origin. Its development forecasts covered June 29–July 14 and July 15–30, 2017, followed by a holdout covering July 31–August 15. It selected by pooled RMSLE on the first two windows. The expanded main study below uses six development windows and three evaluation windows. Later competition iterations reuse these dates with new selection roles, as documented below. [Preliminary run manifest](official-results/manifest.json)
- Measure root mean squared logarithmic error (RMSLE) with one equally weighted term per observation. Clip negative forecasts to zero using the same rule across models, preserve fractional predictions, and keep the exact scoring sample fixed. Summarize each window independently. Compare paired errors by forecast lead, and use MAE in original sales units for business context if available. [Kaggle evaluation](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/overview/evaluation)
- Restrict training, scaling, imputation, series classification, and feature construction to the information available at that fold's origin. For a 16-step forecast, never read realized sales from earlier days of the same forecast window. Planned promotions have a separate interpretation and use only competition-supplied future promotion counts. Actual future oil prices and transactions are unavailable in an operational forecast. Holiday indicators require evidence that the timing was knowable at the origin.

## Preliminary model comparisons

The following comparisons describe the preliminary benchmark. Its results and the subsequent pooled-tree extension motivated the expanded main study.

| Track | Candidates | Scientific comparison |
| --- | --- | --- |
| Sales history | Weekly seasonal naïve, recent weekday mean, Prophet, and t0-beta | Compare methods with observed sales as the common input. Compare Prophet's 0.05 and 0.02 changepoint prior scales to test whether a less flexible short-term trend helps. Tune on development windows only. |
| Planned promotions | Compare Prophet and t0-beta variants with the same future promotion information | Measure whether adding the promotion counts improves prediction on the same folds and rows. This test assumes the historical records reflect the promotion schedule available at each origin. |

Prophet represents a conventional fitted approach, using trend and seasonality components. The Forecasting Company's open-weight `t0-beta` is a decoder-style patch transformer pretrained for time-series forecasts and can condition on known-future covariates. The project uses frozen model weights and logs the checkpoint revision, runtime, context length, device, and point-forecast rule. The model authors report that the beta training corpora excluded fev-bench datasets and disclose that alpha overlapped with Favorita. Favorita appears in fev-bench. This project relies on those disclosures and has not independently audited the complete pretraining corpus, so the transfer interpretation retains that uncertainty. [Prophet](https://github.com/meta/prophet) · [t0-beta model card](https://huggingface.co/theforecastingcompany/t0-beta) · [t0 technical report, §5.1.2](https://arxiv.org/pdf/2609.24559) · [fev-bench dataset](https://github.com/autogluon/fev)

## Expanded evaluation protocol

The original experiment and pooled-tree follow-up compared models on contiguous June–August 2017 windows. The pooled tree was introduced after that study's last window had been examined. The expanded main study fixed the following dates and model configurations before its run. It selects settings using the six development windows, then evaluates the selected specifications on the three later windows. These roles apply to that study. The subsequent calendar and ensemble experiments reuse its evaluation dates for further exploration and selection.

| Role | Train through | Predict |
| --- | --- | --- |
| Development 1 | 2016-06-30 | 2016-07-01 through 2016-07-16 |
| Development 2 | 2016-08-15 | 2016-08-16 through 2016-08-31 |
| Development 3 | 2016-10-31 | 2016-11-01 through 2016-11-16 |
| Development 4 | 2016-12-08 | 2016-12-09 through 2016-12-24 |
| Development 5 | 2017-02-28 | 2017-03-01 through 2017-03-16 |
| Development 6 | 2017-03-31 | 2017-04-01 through 2017-04-16 |
| Evaluation 1 | 2017-04-30 | 2017-05-01 through 2017-05-16 |
| Evaluation 2 | 2017-05-16 | 2017-05-17 through 2017-06-01 |
| Evaluation 3 | 2017-06-12 | 2017-06-13 through 2017-06-28 |

The six development windows span several months and include the same August dates as the competition in the preceding year. Evaluation uses three later 16-day windows whose forecast dates precede the original June–August scoring windows. These are historical, single-retailer estimates. The August 2017 competition file supplies future covariates and test IDs for Kaggle's public scoring period.

Tune the promotion-aware pooled tree using three configurations. The reference uses 43 maximum leaves and an L2 penalty of 1.0. The regularized configuration uses 31 leaves and an L2 penalty of 3.0. The flexible configuration uses 63 leaves and an L2 penalty of 0.5. The learning rate, 150 boosting iterations, and other settings are common across configurations. Select by pooled development RMSLE, weighting each row equally before taking the square root. Fix the selected hyperparameters across the three promotion variants so that changes measure the predictive value of inputs. Compare the development scores of the final candidates to select the competition configuration. The three evaluation windows check how that choice transfers. The experiment script and run manifest record the exact settings and result hashes.

The tree variants use the same sales history, store and family identifiers, and calendar features. The first uses these predictors alone. The second adds trailing promotion counts available at the forecast origin. The third also adds the target day's known-future promotion count. Comparing the second and third variants measures the incremental forecasting value of the supplied schedule under the assumption that it was available at the cutoff. Earlier Prophet and t0-beta variants added past and future promotion counts together, so their historical differences measure the joint contribution. [Pooled tree implementation](../auxiliary/global_benchmark.py) · [Expanded runner](../auxiliary/final_experiment.py)

Compare signed errors and 16-day shortfalls and excesses by window. To show how the choice depends on the ordering objective, apply several assumed ratios of underforecast cost to overforecast cost to those directional errors. These are forecast losses on observed sales. Estimating inventory outcomes calls for reorder constraints, actual inventory, and stockout or waste data.

## Results and provenance

For each run, keep the data source and file hashes, run command, exact date windows, model settings, dependency versions, and execution status. The elapsed seconds in `scores.csv` cover model fitting and forecasting for Prophet and inference for t0-beta. The t0-beta checkpoint is loaded before the timing starts. The notebook documents how to recreate expensive computations. Its primary aggregate bundle is `files/final-results`; `files/official-results` preserves the preliminary benchmark. Calendar and competition follow-ups have separate manifests linked below. Fixture runs support implementation verification. Empirical conclusions use the official competition data. [Course workflow](https://eisenhauerio.github.io/courses-business-decisions/projects/github-workflow.html)

The analysis describes one Ecuadorian retailer and a short historical period. The sales records may reflect availability constraints and do not directly observe demand. Assess performance variation by window, forecast lead, promotion status, or sales volume where supported. Interpret differences in relation to the operational assumptions, compute costs, and available pretraining disclosures.

## Follow-up comparison

After examining the original benchmark, a separate experiment tested a pooled conventional model against the existing Prophet and t0-beta results. The pooled model uses fixed histogram gradient boosting settings and the same 16-day scoring rows. The [follow-up report](global-model-comparison.md) records the experiment, the forecast-origin leakage check, and the retrospective interpretation.

## Calendar follow-up

The competition provides store locations and a calendar of national, regional, and local holidays. It also documents a public-sector salary schedule on the 15th and at month end. These features could improve forecasts if shoppers shift grocery purchases around breaks or payday. The payday indicator encodes information already available in the date, so its comparison tests whether making that calendar pattern explicit improves the tree model. The follow-up uses the frozen promotion-aware HGB configuration and the same nine windows and rows. It compares the reference against adding holidays, payday, and both. The first Kaggle submission uses the HGB model selected by the expanded study's six development windows. [Kaggle data documentation](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data) · [Calendar run manifest](calendar-results/manifest.json)

Regional events match the relevant state and local events match the relevant city. A holiday marked as transferred uses its separately recorded effective date. In the evaluation period, the national Battle of Pichincha holiday moved from May 24 to May 26, 2017. Bridge days and substitute workdays have separate flags. Unscheduled event records are excluded. The official calendar approximates the information available at each cutoff because the files lack historical announcement dates. This experiment reuses the historical evaluation windows after the primary forecast comparisons were examined. Its results show sensitivity to additional predictors rather than evidence for a new competition selection.

## Competition iterations

The first Kaggle submission scored 0.47212 RMSLE on November 16, 2026. The next iteration compared the original HGB and calendar-augmented HGB as blend components. For each, the t0-beta weight took values of 0%, 25%, 50%, 75%, or 100%. The minimum pooled RMSLE over three May–June 2017 windows selected the calendar model with 25% t0-beta. Each blend averages forecasts on the `log1p` scale before converting back to units. [Selection grid](competition-results/blend-weight-scores.csv)

The fixed V2 recipe was then checked on the original June 29–August 15 retrospective benchmark. These were previously examined dates. V1 and V2 forecast August 16–31 from the same cutoff and were scored by Kaggle on all 28,512 test rows. V2's official result was 0.43811 on November 17. The archive records date windows, result hashes, the foundation checkpoint, and actual Kaggle scores. [Historical comparisons](competition-results/backtest.csv) · [Submission record](competition-results/manifest.json)

For V3, four LightGBM specifications were screened on June 13–28, the third of those May–June selection windows. The screen compared their contribution to a fixed 50% calendar HGB700, 25% LightGBM, and 25% t0-beta blend. The product-family model was carried forward. A grid over HGB stages and three-model weights then used all three May–June windows and selected 50% calendar HGB700, 25% product-family LightGBM, and 25% t0-beta. Its pooled selection RMSLE was 0.40067. Holding the recipe fixed on the three previously examined June 29–August 15 windows gave 0.38339, compared with V2's 0.38903. V3 scored 0.42172 on Kaggle on November 19, 2026. [Exploratory screen](refinement-results/screening.csv) · [Historical selection and settings](refinement-results/manifest.json) · [Six-window comparison](refinement-results/backtest.csv) · [V3 submission record](refinement-results/submission-manifest.json)

The August 16–31 competition period is scored in full on one public leaderboard, with no separate private test leaderboard. The three public scores were observed during iterative development, providing feedback for later versions. The recorded gains therefore describe an adaptive sequence of submissions on the same public test period. The historical V2 and V3 comparisons also reuse dates that had already been examined. These results support the documented comparisons under that development history, while future-period performance remains unmeasured. [Kaggle evaluation and leaderboard FAQ](https://www.kaggle.com/c/store-sales-time-series-forecasting)

## Academic transparency

The study grew out of the Store Sales project proposal and the instructor's recommendation to compare conventional forecasts with a time-series foundation model. Codex assisted with course-source review, experiment design, code and tests, benchmark execution and verification, data exploration, and revisions to the documentation and notebook. The implementation, configuration, and artifacts identify which runs produced the empirical results. [Course projects](https://eisenhauerio.github.io/courses-business-decisions/projects/index.html) · [Autumn 2026 course instructions](https://eisenhauerio.github.io/courses-business-decisions/iterations/index.html)

The project-specific engineering includes keyed panel validation, forecast-origin feature construction, calendar alignment, chronological experiment runners, leakage checks, paired error diagnostics, ensemble selection, and reproducible result manifests. The fitted estimators come from scikit-learn, LightGBM, and Prophet. The Forecasting Company supplies the pretrained t0-beta architecture, runtime, and checkpoint. The repository integrates and evaluates these tools and records the assistance above. [Data validation](../auxiliary/data.py) · [Forecast features](../auxiliary/global_benchmark.py) · [Family LightGBM runner](../auxiliary/competition_lightgbm.py) · [Ensemble evaluation](../auxiliary/competition_refinement.py) · [Foundation model adapter](../auxiliary/foundation.py)
