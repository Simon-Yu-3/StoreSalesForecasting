"""Compare a pooled boosted-tree model with the recorded forecasting benchmark."""

import argparse
import json
import os
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from auxiliary.conventional import CONTEXT_LENGTH
from auxiliary.data import Panel, chronological_folds, load_data
from auxiliary.run import errors, submission_frame

ORIGIN_STEP = 15
ROLLING_DAYS = (7, 28, 56, 112)
WEEKDAY_LAGS = (0, 7, 14, 21)
MODEL_SETTINGS = {
    "loss": "squared_error",
    "learning_rate": 0.10,
    "max_iter": 150,
    "max_leaf_nodes": 43,
    "min_samples_leaf": 40,
    "l2_regularization": 1.0,
    "max_bins": 127,
    "early_stopping": False,
    "random_state": 481,
}
MODELS = {"global_hgb": False, "global_hgb_promotion": True}


def origin_features(
    panel: Panel,
    origin: int,
    *,
    promotions: bool,
    historical_promotions: bool | None = None,
) -> pd.DataFrame:
    """Create one row for each series and lead using information at the origin."""
    n_series = len(panel.series)
    leads = np.arange(1, panel.horizon + 1)
    target = origin + leads
    dates = panel.dates[target]
    weekday_anchor = target - 7 * ((leads + 6) // 7)
    store, family = panel.series.codes
    features = {
        "store": np.repeat(store, panel.horizon),
        "family": np.repeat(family, panel.horizon),
        "lead": np.tile(leads, n_series),
        "weekday": np.tile(dates.dayofweek, n_series),
        "day_of_month": np.tile(dates.day, n_series),
        "month": np.tile(dates.month, n_series),
        "month_end": np.tile(dates.is_month_end, n_series),
    }
    for lag in WEEKDAY_LAGS:
        features[f"same_weekday_lag_{lag}"] = np.log1p(
            panel.sales[:, weekday_anchor - lag]
        ).ravel()
    features["last_sales"] = np.repeat(np.log1p(panel.sales[:, origin]), panel.horizon)
    recent = np.log1p(panel.sales[:, origin - max(ROLLING_DAYS) + 1 : origin + 1])
    for days in ROLLING_DAYS:
        trailing = recent[:, -days:]
        features[f"log_mean_{days}"] = np.repeat(
            np.nanmean(trailing, axis=1), panel.horizon
        )
        observed = np.isfinite(trailing).sum(axis=1)
        zero_share = (trailing == 0).sum(axis=1) / observed
        features[f"zero_rate_{days}"] = np.repeat(zero_share, panel.horizon)
    if promotions:
        features["planned_promotion"] = np.log1p(panel.promotions[:, target]).ravel()
    if historical_promotions is None:
        historical_promotions = promotions
    if historical_promotions:
        historical = np.log1p(panel.promotions[:, origin - 27 : origin + 1])
        features["promotion_mean_28"] = np.repeat(
            np.nanmean(historical, axis=1), panel.horizon
        )
    return pd.DataFrame(features, dtype=np.float32)


def training_origins(panel: Panel, cutoff: int) -> range:
    """Return simulated training origins in ascending order within the context."""
    first_day = max(0, cutoff - CONTEXT_LENGTH + 1)
    earliest_origin = first_day + max(ROLLING_DAYS) - 1
    latest_origin = cutoff - panel.horizon
    return range(latest_origin, earliest_origin - 1, -ORIGIN_STEP)[::-1]


def training_examples(
    panel: Panel,
    cutoff: int,
    *,
    promotions: bool,
    historical_promotions: bool | None = None,
):
    """Simulate earlier 16-day forecasts with all labels observed by the cutoff."""
    feature_frames, labels = [], []
    for origin in training_origins(panel, cutoff):
        y = panel.sales[:, origin + 1 : origin + panel.horizon + 1].ravel()
        observed = np.isfinite(y)
        frame = origin_features(
            panel,
            origin,
            promotions=promotions,
            historical_promotions=historical_promotions,
        )
        feature_frames.append(frame.loc[observed])
        labels.append(np.log1p(y[observed]).astype(np.float32))
    if not feature_frames:
        raise ValueError("Insufficient historical windows for the pooled model.")
    return pd.concat(feature_frames, ignore_index=True), np.concatenate(labels)


def fit_predict(panel: Panel, x, y, future, settings):
    """Fit a pooled histogram regressor and return original-unit forecasts."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    model = HistGradientBoostingRegressor(
        categorical_features=["store", "family"], **settings
    )
    model.fit(x, y)
    predicted = np.maximum(np.expm1(model.predict(future)), 0)
    return predicted.reshape(len(panel.series), panel.horizon)


def _predict(panel: Panel, cutoff: int, *, promotions: bool):
    x, y = training_examples(panel, cutoff, promotions=promotions)
    future = origin_features(panel, cutoff, promotions=promotions)
    return fit_predict(panel, x, y, future, MODEL_SETTINGS), len(y)


def _metrics(actual, predicted):
    log_error, absolute_error = errors(actual, predicted)
    series_balance = (predicted - actual).sum(axis=1)
    total_sales = actual.sum()
    return {
        "n": actual.size,
        "rmsle": float(np.sqrt(log_error.mean())),
        "mae": float(absolute_error.mean()),
        "wape": float(absolute_error.sum() / total_sales) if total_sales else None,
        "net_bias_units": float(series_balance.sum()),
        "horizon_underforecast_units": float(np.maximum(-series_balance, 0).sum()),
        "horizon_overforecast_units": float(np.maximum(series_balance, 0).sum()),
    }


def main():
    import sklearn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--prediction-model",
        choices=MODELS,
        help="Refit this configuration and write Kaggle test forecasts only.",
    )
    args = parser.parse_args()

    panel = load_data(args.data)
    if args.prediction_model is not None:
        cutoff = panel.dates.get_loc(panel.train_end)
        dates = panel.dates[panel.dates > panel.train_end]
        prediction, _ = _predict(
            panel, cutoff, promotions=MODELS[args.prediction_model]
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        submission_frame(panel, dates, prediction).to_csv(
            args.output_dir / "submission.csv", index=False
        )
        return
    score_rows = []
    for fold in chronological_folds(panel):
        cutoff = panel.dates.get_loc(fold.train_end)
        actual = panel.sales[:, panel.dates.get_indexer(fold.forecast_dates)]
        for name, use_promotions in MODELS.items():
            start = perf_counter()
            prediction, n_training = _predict(panel, cutoff, promotions=use_promotions)
            metrics = _metrics(actual, prediction)
            score_rows.append(
                {
                    "fold": fold.name,
                    "model": name,
                    "train_end": str(fold.train_end.date()),
                    "forecast_start": str(fold.forecast_dates[0].date()),
                    "forecast_end": str(fold.forecast_dates[-1].date()),
                    "training_rows": n_training,
                    "seconds": perf_counter() - start,
                    **metrics,
                }
            )
            print(f"{fold.name} {name}: RMSLE {metrics['rmsle']:.5f}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(score_rows).to_csv(args.output_dir / "scores.csv", index=False)
    record = {
        "data": panel.provenance,
        "settings": MODEL_SETTINGS,
        "origin_stride_days": ORIGIN_STEP,
        "context_days": CONTEXT_LENGTH,
        "horizon": panel.horizon,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "omp_threads": os.environ.get("OMP_NUM_THREADS"),
        },
        "evaluation": "Retrospective extension on the original benchmark windows.",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
