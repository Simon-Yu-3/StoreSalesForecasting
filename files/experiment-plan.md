# Experiment plan

## Question and decision

Which methods produce the most accurate 16-day-ahead daily sales forecasts for the typical store–product-family observation at Corporación Favorita? How does the ranking change when planners can supply scheduled promotions? The decision is which forecasting process to use as an input to short-run replenishment or staffing. Forecast accuracy under Kaggle's loss does not measure stockouts, profit, or the causal effect of a promotion. The project follows the course's [Decision Loop](https://eisenhauerio.github.io/courses-business-decisions/decision-loop/index.html) by connecting a measurable forecast objective to a decision and a repeatable evaluation.

On October 23, 2026, Professor Philipp Eisenhauer approved the Store Sales topic and asked for a comparison of conventional methods and LLM-based time-series foundation models. He suggested Prophet and The Forecasting Company as starting points. This paraphrase records the approval without republishing the private correspondence.

## Evaluation

- The observation is daily unit sales for one store–family pair. Sales can be fractional. Run all methods on the same keyed forecast rows and same date cutoffs. Read and check the horizon against the official test dates before a competition fit. [Competition data](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data)
- Use rolling-origin backtests with up to 512 calendar days of history at each origin. Select the lowest pooled RMSLE across the first two 16-day development windows. Score prespecified variants on the last labeled window after model selection. Apply the development-selected variant at the final labeled date under the same context limit for optional competition predictions, which have no local ground truth.
- Measure root mean squared logarithmic error (RMSLE) with one equally weighted term per observation. Clip negative forecasts to zero using the same rule across models, preserve fractional predictions, and keep the exact scoring sample fixed. Summarize each window independently. Compare paired errors by forecast lead, and use MAE in original sales units for business context if available. [Kaggle evaluation](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/overview/evaluation)
- Restrict training, scaling, imputation, series classification, and feature construction to the information available at that fold's origin. For a 16-step forecast, never read realized sales from earlier days of the same forecast window. Planned promotions have a separate interpretation and use only competition-supplied future promotion counts. Actual future oil prices and transactions are unavailable in an operational forecast. Holiday indicators require evidence that the timing was knowable at the origin.

## Model comparisons

| Track | Candidates | Scientific comparison |
| --- | --- | --- |
| Sales history | Weekly seasonal naïve, recent weekday mean, Prophet, and t0-beta | Compare methods with observed sales as the common input. Compare Prophet's 0.05 and 0.02 changepoint prior scales to test whether a less flexible short-term trend helps. Tune on development windows only. |
| Planned promotions | Compare Prophet and t0-beta variants with the same future promotion information | Measure whether adding the promotion counts improves prediction on the same folds and rows. This test assumes the historical records reflect the promotion schedule available at each origin. |

Prophet represents a conventional fitted approach, using trend and seasonality components. The Forecasting Company's open-weight `t0-beta` is a decoder-style patch transformer pretrained for time-series forecasts and can condition on known-future covariates. Use frozen model weights and log the checkpoint revision, runtime, context length, device, and point-forecast rule. The model authors report that the beta training corpora excluded fev-bench datasets and disclose that alpha overlapped with Favorita. Favorita appears in fev-bench, so verify overlap for any further checkpoint before interpreting transfer performance. [Prophet](https://github.com/meta/prophet) · [t0-beta model card](https://huggingface.co/theforecastingcompany/t0-beta) · [t0 technical report, §5.1.2](https://arxiv.org/pdf/2609.24559) · [fev-bench dataset](https://github.com/autogluon/fev)

## Results and provenance

For each run, keep the data source and file hashes, run command, exact date windows, model settings, dependency versions, and execution status. The elapsed seconds in `scores.csv` cover model fitting and forecasting for Prophet and inference for t0-beta. The t0-beta checkpoint is loaded before the timing starts. The notebook documents how to recreate expensive computations. It can read a compact bundle of aggregate results from `files/official-results` if the competition rules permit publishing those artifacts. Fixture runs support implementation verification. Report empirical conclusions using the official competition data. [Course workflow](https://eisenhauerio.github.io/courses-business-decisions/projects/github-workflow.html)

The analysis describes one Ecuadorian retailer and a short historical period. The sales records may reflect availability constraints and do not directly observe demand. Assess performance variation by window, forecast lead, promotion status, or sales volume where supported. Interpret differences in relation to the operational assumptions, compute costs, and available pretraining disclosures.

## Follow-up comparison

After examining the original benchmark, a separate experiment tested a pooled conventional model against the existing Prophet and t0-beta results. The pooled model uses fixed histogram gradient boosting settings and the same 16-day scoring rows. The [follow-up report](global-model-comparison.md) records the experiment, the forecast-origin leakage check, and the retrospective interpretation.

## Academic transparency

Codex assisted with course-source review, benchmark design, implementation, tests, and draft documentation. Review the final artifacts and record the student's actual contributions and interpretation before submission. Identify results with their generating runs and cite borrowed methods, libraries, and data. The course's public guidance records the project and submission workflow. Check any additional course-specific disclosure directions in the student repository or course communications. [Course projects](https://eisenhauerio.github.io/courses-business-decisions/projects/index.html) · [Autumn 2026 course instructions](https://eisenhauerio.github.io/courses-business-decisions/iterations/index.html)
