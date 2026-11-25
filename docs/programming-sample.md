# Programming sample

[Two-page code PDF](../output/pdf/StoreSalesForecasting-Programming-Supplement.pdf) · [Executable Python](../examples/programming_sample.py) · [Research notebook](../project.ipynb)

This sample makes the project's programming work directly inspectable. A short script reads CSV data, checks its schema, constructs a calendar-aligned lag, fits a gradient-boosted tree, evaluates a complete 16-day forecast, and writes a metrics table and plot. The PDF extracts complete source functions and selected output lines from that script. Line numbers refer to the executable file.

## Run the example

From the repository root after installing the package:

```bash
python -m pip install -e ".[test]"
python examples/programming_sample.py --fixture
python -m pytest tests/test_programming_sample.py tests/test_submission_contract.py -q
```

The default generated data cover six store-family series and deliberately omit one calendar date. The output files are `results/programming-sample/metrics.csv` and `forecast.png`. They demonstrate execution on generated data. The scored competition ensemble is documented separately in the research report.

The same script accepts the original Kaggle archive:

```bash
python examples/programming_sample.py \
  --data data/raw/store-sales-time-series-forecasting.zip \
  --output-dir results/programming-sample-official
```

The compact model evaluates the final 16 labeled days in `train.csv`. Its hyperparameters are fixed in the example. The full competition ensemble has richer features, separately selected weights, and its own historical comparisons.

## What to inspect

| Programming task | Implementation | Verification |
| --- | --- | --- |
| CSV ingestion and data types | `read_sales` parses dates, strings, integers, and fractional sales | Schema, duplicate-key, negative-value, and malformed-number tests |
| Data wrangling | `build_features` joins sales on the date exactly 28 days earlier and encodes categorical fields | A missing-calendar-day regression test |
| Custom analysis functions | `forecast` calls the feature builder, fits on dates through one cutoff, and forecasts the full horizon | Replacing every held-out target leaves all 16 forecasts unchanged |
| Visualization and metrics | `evaluate` calculates RMSLE and MAE, groups results by date, and exports a plot and CSV | Both generated and official-data commands were executed |
| Full-project forecasting | [HGB features](../auxiliary/global_benchmark.py), [family LightGBM](../auxiliary/competition_lightgbm.py), [blend selection](../auxiliary/competition_refinement.py) | Native HGB and LightGBM integration runs in [CI](../.github/workflows/programming-smoke.yml) |
| Submission integrity | [Keyed export](../auxiliary/run.py) requires exact matrix shape and complete, unique test keys | Reordered dates and rows preserve their predictions; ambiguous grids are rejected |

The lag uses a date-keyed join because removing December 25 should leave its 28-day counterpart missing. A 28-row shift would choose the wrong date after a calendar gap. At forecast lead 16, the lagged sale occurred 12 days before the cutoff, so every lag value is already observable when the entire forecast is issued.

The native integration test fits actual HGB and LightGBM estimators on a small generated panel. A seasonal baseline supplies its third component. The production t0-beta checkpoint is integrated and evaluated separately.

## Research result and interpretation

The best recorded Kaggle submission scored **0.42172 RMSLE**, a **10.68%** reduction from the first submission's **0.47212** on the same August 16-31, 2017 test period. The selected blend uses 50% calendar HGB700, 25% family LightGBM, and 25% promotion-aware t0-beta on log sales. The [submission record](../files/refinement-results/submission-manifest.json) connects the score evidence to the CSV checksum and frozen historical recipe.

Three public Kaggle scores were observed during iterative development. Later historical comparisons also reuse previously reviewed dates. The documented results describe that development history; future-period accuracy and inventory or profit effects have not been measured. [Experiment chronology and limitations](../files/experiment-plan.md)

## Code and model provenance

This course project used AI-assisted coding, including Codex, across implementation, tests, experiments, and documentation. Project-specific engineering includes data validation, date and feature alignment, forecast runners, error diagnostics, and artifact checks. Scikit-learn, LightGBM, and Prophet provide the estimators. The Forecasting Company provides the pretrained t0-beta model and checkpoint. [Full assistance and model-source disclosure](../files/experiment-plan.md#academic-transparency)

New tree-model caches record a feature schema version and Python source hashes. Earlier checked-in results retain their original manifests. The pinned research environments and commands are in the [reproduction guide](reproduction.md).

## Regenerate the PDF

```bash
python -m pip install -e ".[supplement]"
python examples/programming_sample.py --fixture
python tools/build_programming_supplement.py
```

The builder reads the actual Python source and generated example output. It checks code width and footer clearance, and writes a two-page PDF with selectable code and a linked repository footer.
