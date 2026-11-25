"""Run chronological backtests and write reproducible forecast summaries."""

import argparse
import csv
import importlib.metadata
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from auxiliary.conventional import CONTEXT_LENGTH, prophet_forecast, weekday_forecast
from auxiliary.data import chronological_folds, load_data, make_fixture

BASELINES = ("seasonal_naive", "weekday_mean")
CORE_MODELS = ("seasonal_naive", "prophet", "t0_beta")
MODELS = (
    *BASELINES,
    "prophet",
    "prophet_rigid",
    "prophet_promotion",
    "t0_beta",
    "t0_beta_promotion",
)


def errors(actual, predicted):
    """Return per-observation squared log error and absolute unit error."""
    if actual.shape != predicted.shape:
        raise ValueError("Forecast shape does not match the target panel.")
    if not (np.isfinite(actual).all() and np.isfinite(predicted).all()):
        raise ValueError("Scoring requires a forecast and an actual on every row.")
    if (actual < 0).any() or (predicted < 0).any():
        raise ValueError("RMSLE requires nonnegative actuals and forecasts.")
    return np.square(np.log1p(predicted) - np.log1p(actual)), np.abs(predicted - actual)


def _metrics(log_error, absolute_error):
    return {
        "n": log_error.size,
        "rmsle": float(np.sqrt(log_error.mean())),
        "mae": float(absolute_error.mean()),
    }


def _slice_scores(fold, model, log_error, abs_error, promotions):
    for step in range(log_error.shape[1]):
        yield {
            "fold": fold,
            "model": model,
            "dimension": "horizon_day",
            "group": str(step + 1),
            **_metrics(log_error[:, step], abs_error[:, step]),
        }
    for label, mask in (
        ("no_promotion", promotions == 0),
        ("promotion", promotions > 0),
    ):
        if mask.any():
            yield {
                "fold": fold,
                "model": model,
                "dimension": "promotion",
                "group": label,
                **_metrics(log_error[mask], abs_error[mask]),
            }


def _forecast(panel, model, train_end, forecast_dates, foundation, jobs):
    history_end = panel.dates.get_loc(train_end) + 1
    history_start = max(0, history_end - CONTEXT_LENGTH)
    history_dates = panel.dates[history_start:history_end]
    history = panel.sales[:, history_start:history_end]
    if model in BASELINES:
        return weekday_forecast(
            history, history_dates, forecast_dates, average=(model == "weekday_mean")
        )

    promo = {}
    if model.endswith("_promotion"):
        future_indices = panel.dates.get_indexer(forecast_dates)
        if (future_indices < history_end).any():
            raise ValueError(
                "Promotion inputs must align to the future forecast dates."
            )
        promo = {
            "historical_promo": panel.promotions[:, history_start:history_end],
            "future_promo": panel.promotions[:, future_indices],
        }

    if model.startswith("prophet"):
        return prophet_forecast(
            history,
            history_dates,
            forecast_dates,
            jobs=jobs,
            changepoint_prior_scale=0.02 if model == "prophet_rigid" else 0.05,
            **promo,
        )
    return foundation.predict(history, len(forecast_dates), **promo)


def _submission(panel, model, foundation, jobs):
    dates = panel.dates[panel.dates > panel.train_end]
    predictions = _forecast(panel, model, panel.train_end, dates, foundation, jobs)
    return submission_frame(panel, dates, predictions)


def submission_frame(panel, dates, predictions):
    """Map a complete forecast matrix to the original Kaggle test IDs."""
    dates = pd.DatetimeIndex(dates)
    predictions = np.asarray(predictions)
    if (
        len(dates) != panel.horizon
        or not dates.is_unique
        or dates.hasnans
        or not panel.series.is_unique
    ):
        raise ValueError("Submission requires unique series and all forecast dates.")
    expected_shape = (len(panel.series), panel.horizon)
    if predictions.shape != expected_shape:
        raise ValueError(f"Submission forecast shape must be {expected_shape}.")
    if not np.isfinite(predictions).all() or (predictions < 0).any():
        raise ValueError("Submission must contain finite nonnegative forecasts.")
    test = panel.test
    keys = ["date", "store_nbr", "family"]
    if (
        len(test) != expected_shape[0] * expected_shape[1]
        or test["id"].isna().any()
        or not test["id"].is_unique
        or test[keys].isna().any().any()
        or test.duplicated(keys).any()
    ):
        raise ValueError("Test IDs must identify a complete, unique forecast grid.")
    store_family = pd.MultiIndex.from_frame(test[["store_nbr", "family"]])
    series_index = panel.series.get_indexer(store_family)
    date_index = dates.get_indexer(test["date"])
    if (series_index < 0).any() or (date_index < 0).any():
        raise ValueError("Could not align every test ID to a forecast.")
    return pd.DataFrame(
        {"id": test["id"], "sales": predictions[series_index, date_index]}
    )


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _package_versions(models, device):
    packages = ["numpy", "pandas"]
    if any(model.startswith("prophet") for model in models):
        packages.extend(("prophet", "cmdstanpy"))
    if any(model.startswith("t0") for model in models):
        packages.extend(
            ("tfc-t0", "huggingface-hub", "mlx" if device == "mlx" else "torch")
        )
    return {name: importlib.metadata.version(name) for name in packages}


def run(
    panel, models, output_dir, *, fixture, device, batch_size, jobs, make_submission
):
    """Evaluate fixed candidate configurations and select on development windows."""
    foundation = None
    model_metadata = None
    if any(model.startswith("t0") for model in models):
        from auxiliary.foundation import MODEL_ID, MODEL_REVISION, T0Forecaster

        load_start = perf_counter()
        foundation = T0Forecaster(batch_size=batch_size, device=device)
        model_metadata = {
            "checkpoint": MODEL_ID,
            "revision": MODEL_REVISION,
            "batch_size": batch_size,
            "load_seconds": perf_counter() - load_start,
        }

    scores, slices, fold_metadata = [], [], []
    dev_count = dict.fromkeys(models, 0)
    dev_squared_log_error = dict.fromkeys(models, 0.0)
    folds = chronological_folds(panel)
    selected_model = None
    for fold in folds:
        if fold.name == "holdout":
            dev_rmsle = {
                model: float(np.sqrt(dev_squared_log_error[model] / dev_count[model]))
                for model in models
            }
            selected_model = min(models, key=dev_rmsle.get)
        identity = {
            "fold": fold.name,
            "train_end": fold.train_end.date().isoformat(),
            "forecast_start": fold.forecast_dates[0].date().isoformat(),
            "forecast_end": fold.forecast_dates[-1].date().isoformat(),
        }
        fold_metadata.append(identity)
        indices = panel.dates.get_indexer(fold.forecast_dates)
        if (indices < 0).any() or (fold.forecast_dates <= fold.train_end).any():
            raise ValueError(
                "Forecast dates must be present and strictly after the cutoff."
            )
        for model in models:
            start = perf_counter()
            prediction = _forecast(
                panel, model, fold.train_end, fold.forecast_dates, foundation, jobs
            )
            seconds = perf_counter() - start
            log_error, abs_error = errors(panel.sales[:, indices], prediction)
            if fold.name != "holdout":
                dev_count[model] += log_error.size
                dev_squared_log_error[model] += float(log_error.sum())
            row = {
                **identity,
                "model": model,
                **_metrics(log_error, abs_error),
                "seconds": seconds,
            }
            scores.append(row)
            slices.extend(
                _slice_scores(
                    fold.name, model, log_error, abs_error, panel.promotions[:, indices]
                )
            )
            print(
                f"{fold.name:12} {model:20} RMSLE={row['rmsle']:.5f} ({seconds:.2f}s)",
                flush=True,
            )

    if selected_model is None:
        raise ValueError("Expected a holdout after the development windows.")
    submission = None
    if make_submission:
        submission = _submission(panel, selected_model, foundation, jobs)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest_path.unlink(missing_ok=True)
    (output_dir / "submission.csv").unlink(missing_ok=True)
    _write_csv(output_dir / "scores.csv", scores)
    _write_csv(output_dir / "slices.csv", slices)
    if submission is not None:
        submission.to_csv(output_dir / "submission.csv", index=False)
    manifest = {
        "dataset": "fixture" if fixture else "kaggle_store_sales",
        "data": panel.provenance,
        "n_series": len(panel.series),
        "horizon": panel.horizon,
        "models": list(models),
        "python": platform.python_version(),
        "packages": _package_versions(models, device),
        "device": device,
        "jobs": jobs,
        "context_length": CONTEXT_LENGTH,
        "foundation_model": model_metadata,
        "folds": fold_metadata,
        "development_rmsle": dev_rmsle,
        "selected_model": selected_model,
        "submission_generated": make_submission,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--fixture", action="store_true", help="Use synthetic test data."
    )
    source.add_argument(
        "--data", type=Path, help="Official Kaggle ZIP or folder of CSVs."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS)
    parser.add_argument(
        "--device", choices=("cpu", "mlx", "mps", "cuda"), default="cpu"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--jobs", type=int, default=1, help="Concurrent Prophet fits.")
    parser.add_argument("--make-submission", action="store_true")
    args = parser.parse_args()
    models = tuple(args.models or (BASELINES if args.fixture else MODELS))
    if args.jobs < 1 or args.batch_size < 1 or len(set(models)) != len(models):
        parser.error("Use unique models and positive --jobs and --batch-size.")
    if args.make_submission and (args.fixture or not set(CORE_MODELS) <= set(models)):
        parser.error(
            "--make-submission requires official data and all three core models."
        )
    panel = make_fixture() if args.fixture else load_data(args.data)
    run(
        panel,
        models,
        args.output_dir,
        fixture=args.fixture,
        device=args.device,
        batch_size=args.batch_size,
        jobs=args.jobs,
        make_submission=args.make_submission,
    )


if __name__ == "__main__":
    main()
