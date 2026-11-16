# Forecasting store sales with promotion schedules

**UW ECON 481A · Autumn 2026**

A replenishment planner needs 16 days of forecasts before the next ordering cycle. For Ecuadorian grocer Corporación Favorita, we compare seasonal baselines, independently fitted Prophet models, a pooled gradient-boosted tree, and the pretrained t0-beta time-series model. Within the pooled tree, we test the incremental predictive value of a future promotion schedule after accounting for prior promotions.

[Executed project notebook](project.ipynb) · [Experiment protocol and credits](files/experiment-plan.md) · [Result manifest](files/final-results/manifest.json) · [Reproduce](#reproduce-the-analysis)

## Findings

The development-selected pooled tree with scheduled promotions had the lowest RMSLE in all three evaluation windows. Its pooled evaluation RMSLE was **0.41636**, about **2.2%** lower than the promotion-aware t0-beta model. t0-beta had the lowest daily MAE and WAPE in each evaluation window. The preferred model therefore depends on whether the goal is competition accuracy or fewer unit errors in replenishment.

| Configuration | Development RMSLE | Evaluation RMSLE | Evaluation MAE |
| --- | ---: | ---: | ---: |
| HGB · past and scheduled promotions | **0.52069** | **0.41636** | 71.06 |
| HGB · sales and calendar | 0.53746 | 0.42524 | 71.26 |
| t0-beta · past and scheduled promotions | 0.52523 | 0.42565 | **66.59** |
| HGB · past promotions only | 0.53844 | 0.42662 | 71.77 |
| Prophet · past and scheduled promotions | 0.67694 | 0.50242 | 128.24 |
| Seasonal naïve | 0.67618 | 0.56105 | 87.68 |
| Eight-week weekday mean | 0.64981 | 0.59415 | 78.42 |

**The value of the schedule.** With HGB settings fixed, adding the schedule to past promotions reduced pooled evaluation RMSLE from **0.42662 to 0.41636** and reduced RMSLE in all three windows. Total 16-day underforecast units fell in two windows and rose by about **44,000 units** in the window starting May 17. The notebook compares both directional errors under several hypothetical cost ratios. [Overall scores](files/final-results/scores.csv) · [Lead and group scores](files/final-results/slices.csv)

## Data and evaluation

| Item | Coverage |
| --- | --- |
| Forecast unit | Daily sales at one store for one product family |
| Training data | 3,000,888 observations across 54 stores, 33 families, and January 2013 to August 2017 |
| Historical comparison | Six 16-day development windows and three later 16-day evaluation windows |
| Kaggle test | 28,512 rows for August 16–31, 2017 |

All models issue the full 16-day forecast from each cutoff. Sales features use up to 512 calendar days through that cutoff. For the pooled model, training examples simulate earlier forecast origins whose targets were already observed. A regression test changes all post-cutoff sales and checks that features and training labels are identical. Another checks that changing future promotions affects only the model allowed to read them. [Competition data](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data) · [Evaluation protocol](files/experiment-plan.md#expanded-evaluation-protocol)

RMSLE is the competition metric. We pool the squared log errors at the row level before taking the square root. We report each evaluation window separately and also track MAE in sales units and WAPE as a percentage of observed sales. The three HGB variants share the same tree settings to measure the value of the inputs under comparable conditions. The development windows select the tree settings and the competition model. Evaluation windows assess how well that selection transfers.

The future promotion counts are supplied for the Kaggle test, and the historical analysis assumes a planner would have known those values by each cutoff. The dataset does not record schedule revisions or unconstrained customer demand. The inventory discussion uses directional errors on observed sales and hypothetical cost weights. The next operational validation is a parallel pilot using saved schedules, actual inventory, and ordering constraints.

## Kaggle submission

The final runner refits HGB with scheduled promotions and exports 28,512 nonnegative predictions aligned to the sample submission IDs. Its submission manifest records the model, test dates, and prediction file hash. An official Kaggle score is pending. [Competition evaluation](https://www.kaggle.com/c/store-sales-time-series-forecasting)

## Reproduce the analysis

### Execute the report and verify the implementation

```bash
conda env create -f environment.yml
conda activate econ481-store-sales
python -m pytest tests
python -m auxiliary.final_experiment --fixture --output-dir results/final-fixture
ECON481_RESULTS=results/final-fixture python -m pytest --nbmake project.ipynb
ECON481_RESULTS=files/final-results python -m pytest --nbmake project.ipynb
```

The report reads one result directory per execution. By default it reads the checked-in aggregate bundle. Set `ECON481_RESULTS` to analyze a rerun. GitHub Actions runs Ruff, Python tests and both notebook executions. The small fixture verifies the pipeline, and saved official aggregates keep the default check independent of competition authentication and checkpoint downloads. [Workflow](.github/workflows/run-notebook.yml)

### Regenerate the official results

Download the [Kaggle archive](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data) to `data/raw/store-sales-time-series-forecasting.zip`. In the course environment, install the inference dependencies. Linux CPU runners can install `torch==2.15.0` from the [CPU wheel index](https://download.pytorch.org/whl/cpu) first.

```bash
python -m pip install -r requirements-t0.txt
export OMP_NUM_THREADS=4
data=data/raw/store-sales-time-series-forecasting.zip
cache=results/final-model-cache

python -m auxiliary.final_experiment --data "$data" --output-dir "$cache" \
  --model-shard prophet_promotion --jobs 8
python -m auxiliary.final_experiment --data "$data" --output-dir "$cache" \
  --model-shard t0_beta_promotion --device cpu
python -m auxiliary.final_experiment --data "$data" --output-dir results/final \
  --prediction-cache "$cache"
ECON481_RESULTS=results/final python -m pytest --nbmake project.ipynb
python -m auxiliary.final_experiment --data "$data" --output-dir results/kaggle \
  --submission-from results/final/manifest.json
```

The shard commands run the slow models in separate Python processes. The aggregate run validates the cache against the data hashes and cutoff dates before training and comparing the pooled model. With competition access, the optional `full_experiment` GitHub Actions dispatch regenerates the report tables using a repository `KAGGLE_API_TOKEN` secret.

## Earlier experiments

The earlier contiguous June–August 2017 comparison and retrospective pooled extension remain available as a record of the project's development. [Original benchmark](files/official-results/scores.csv) · [Pooled model follow-up](files/global-model-comparison.md) · [Earlier dashboard](files/forecast-dashboard.html) · [Earlier report](files/store-sales-research-report.pdf)

The repository follows the [ECON 481A student project template](https://github.com/eisenhauerIO/projects-student-template).
