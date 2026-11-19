"""Fit a direct 16-day LightGBM forecast at one strictly historical origin."""

import argparse
import gc
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from auxiliary.calendar_experiment import _load_calendar, _target_flags, calendar_flags
from auxiliary.data import Panel, load_data
from auxiliary.global_benchmark import _metrics, origin_features
from auxiliary.run import submission_frame

CATEGORICAL = (
    "store",
    "family",
    "store_family",
    "city",
    "state",
    "store_type",
    "cluster",
)
CONFIGURATIONS = {
    "compact": {
        "n_estimators": 600,
        "learning_rate": 0.04,
        "num_leaves": 63,
        "min_child_samples": 80,
        "reg_lambda": 1.0,
        "reg_alpha": 0.05,
    },
    "deep": {
        "n_estimators": 900,
        "learning_rate": 0.035,
        "num_leaves": 127,
        "min_child_samples": 100,
        "reg_lambda": 2.0,
        "reg_alpha": 0.05,
    },
    "family": {
        "n_estimators": 500,
        "learning_rate": 0.04,
        "num_leaves": 31,
        "min_child_samples": 40,
        "reg_lambda": 3.0,
        "reg_alpha": 0.05,
    },
}
LOOKBACK = 371


def _store_features(panel, stores):
    ordered = stores.set_index("store_nbr").loc[panel.series.levels[0]]
    store_code = panel.series.codes[0]
    output = {"store_family": np.arange(len(panel.series), dtype=np.float32)}
    for column, name in (
        ("city", "city"),
        ("state", "state"),
        ("type", "store_type"),
        ("cluster", "cluster"),
    ):
        code = pd.factorize(ordered[column], sort=True)[0]
        output[name] = code[store_code].astype(np.float32)
    return {name: np.repeat(values, panel.horizon) for name, values in output.items()}


def features_at_origin(panel: Panel, origin: int, flags, store_features):
    """Use observed sales through origin and promotions through origin + horizon."""
    if origin < LOOKBACK:
        raise ValueError("Feature construction requires 371 past calendar days.")
    n_series = len(panel.series)
    horizon = panel.horizon
    leads = np.arange(1, horizon + 1)
    targets = origin + leads
    anchors = targets - 7 * ((leads + 6) // 7)
    base = origin_features(panel, origin, promotions=True)
    calendar = _target_flags(panel, origin, flags).astype("float32")
    extra = dict(store_features)

    def daily(name, values):
        extra[name] = np.asarray(values, dtype=np.float32).ravel()

    def fixed(name, values):
        extra[name] = np.repeat(np.asarray(values, dtype=np.float32), horizon)

    recent = panel.sales[:, origin - 111 : origin + 1]
    weekday_history = np.log1p(panel.sales[:, anchors[:, None] - 7 * np.arange(8)])
    for weeks in (2, 4, 8):
        daily(
            f"weekday_log_mean_{weeks}",
            np.nanmean(weekday_history[:, :, :weeks], axis=2),
        )
    daily("weekday_log_std_8", np.nanstd(weekday_history, axis=2))
    daily(
        "weekday_recent_shift",
        weekday_history[:, :, 0] - np.nanmean(weekday_history[:, :, :4], axis=2),
    )
    for lag in (28, 56, 364):
        daily(f"target_lag_{lag}", np.log1p(panel.sales[:, targets - lag]))
    for lag in (1, 2, 3, 6, 13, 27):
        fixed(f"origin_sales_lag_{lag}", np.log1p(panel.sales[:, origin - lag]))

    raw_means = {}
    for days in (3, 7, 14, 28):
        raw_means[days] = np.nanmean(recent[:, -days:], axis=1)
        fixed(f"raw_log_mean_{days}", np.log1p(raw_means[days]))
    fixed("raw_log_std_28", np.log1p(np.nanstd(recent[:, -28:], axis=1)))
    fixed("weekly_level_shift", np.log1p(raw_means[7]) - np.log1p(raw_means[28]))
    fixed("last_three_level_shift", np.log1p(raw_means[3]) - np.log1p(raw_means[28]))
    positive = recent[:, -28:] > 0
    count = positive.sum(axis=1)
    positive_sum = np.where(positive, recent[:, -28:], 0).sum(axis=1)
    positive_mean = np.divide(
        positive_sum, count, out=np.zeros(n_series), where=count > 0
    )
    fixed("positive_sales_log_mean_28", np.log1p(positive_mean))
    has_sales = recent > 0
    last_positive_age = np.where(
        has_sales.any(axis=1), has_sales[:, ::-1].argmax(axis=1), 112
    )
    fixed("days_since_positive_sale", last_positive_age)

    future_promotion = np.log1p(panel.promotions[:, targets])
    promotion_history = np.log1p(panel.promotions[:, origin - 27 : origin + 1])
    daily(
        "promotion_weekday_change",
        future_promotion - np.log1p(panel.promotions[:, anchors]),
    )
    daily(
        "promotion_level_change",
        future_promotion - np.nanmean(promotion_history, axis=1)[:, None],
    )
    fixed("promotion_log_mean_7", np.nanmean(promotion_history[:, -7:], axis=1))
    fixed(
        "promotion_weekly_shift",
        np.nanmean(promotion_history[:, -7:], axis=1)
        - np.nanmean(promotion_history, axis=1),
    )
    fixed("promotion_horizon_log_mean", np.mean(future_promotion, axis=1))
    for offset, name in ((-1, "promotion_previous_plan"), (1, "promotion_next_plan")):
        neighbors = np.clip(np.arange(horizon) + offset, 0, horizon - 1)
        daily(name, future_promotion[:, neighbors])

    n_stores, n_families = [len(level) for level in panel.series.levels]
    for days in (7, 28):
        means = raw_means[days].reshape(n_stores, n_families)
        store_level = np.broadcast_to(
            np.mean(means, axis=1, keepdims=True), means.shape
        )
        family_level = np.broadcast_to(
            np.mean(means, axis=0, keepdims=True), means.shape
        )
        fixed(f"store_log_mean_{days}", np.log1p(store_level.ravel()))
        fixed(f"family_log_mean_{days}", np.log1p(family_level.ravel()))
    dates = panel.dates[targets]
    day_angle = dates.dayofyear.to_numpy() * 2 * np.pi / 365.25
    daily("dayofyear_sin", np.broadcast_to(np.sin(day_angle), (n_series, horizon)))
    daily("dayofyear_cos", np.broadcast_to(np.cos(day_angle), (n_series, horizon)))
    daily("year", np.broadcast_to(dates.year.to_numpy(), (n_series, horizon)))
    extras = pd.DataFrame(extra, dtype=np.float32)
    return pd.concat((base, calendar, extras), axis=1)


def training_origins(panel, cutoff, *, history_days, origin_step, max_origins):
    if (
        cutoff < 0
        or cutoff >= len(panel.dates)
        or panel.dates[cutoff] > panel.train_end
    ):
        raise ValueError("Training cutoff must be an observed official training date.")
    earliest = max(LOOKBACK, cutoff - history_days + 112)
    origins = list(range(cutoff - panel.horizon, earliest - 1, -origin_step))[
        :max_origins
    ]
    if not origins:
        raise ValueError("No complete simulated forecasts precede the cutoff.")
    return origins[::-1]


def build_training(panel, cutoff, flags, store_features, args):
    origins = training_origins(
        panel,
        cutoff,
        history_days=args.history_days,
        origin_step=args.origin_step,
        max_origins=args.max_origins,
    )
    future = features_at_origin(panel, cutoff, flags, store_features)
    columns = list(future.columns)
    observed_masks = [
        np.isfinite(panel.sales[:, origin + 1 : origin + panel.horizon + 1]).ravel()
        for origin in origins
    ]
    rows = sum(mask.sum() for mask in observed_masks)
    x = np.empty((rows, len(columns)), dtype=np.float32)
    y = np.empty(rows, dtype=np.float32)
    weights = np.ones(rows, dtype=np.float32)
    position = 0
    for origin, observed in zip(origins, observed_masks, strict=True):
        if origin + panel.horizon > cutoff:
            raise ValueError("A simulated forecast label extends beyond the cutoff.")
        features = features_at_origin(panel, origin, flags, store_features)
        size = int(observed.sum())
        x[position : position + size] = features.to_numpy(copy=False)[observed]
        target = panel.sales[:, origin + 1 : origin + panel.horizon + 1].ravel()[
            observed
        ]
        y[position : position + size] = np.log1p(target)
        if args.half_life_days:
            age = cutoff - (origin + panel.horizon)
            weights[position : position + size] = 2 ** (-age / args.half_life_days)
        position += size
    del observed_masks
    gc.collect()
    return pd.DataFrame(x, columns=columns, copy=False), y, weights, future, origins


def fit_predictions(x, y, weights, future, settings, *, by_family):
    """Fit the shared specification globally or in sequential family groups."""
    import lightgbm as lgb

    groups = sorted(future["family"].unique()) if by_family else [None]
    prediction = np.empty(len(future), dtype=np.float64)
    gain = np.zeros(len(x.columns), dtype=np.float64)
    fitted_groups = []
    for index, family in enumerate(groups, 1):
        train_rows = (
            slice(None) if family is None else x["family"].eq(family).to_numpy()
        )
        future_rows = (
            slice(None) if family is None else future["family"].eq(family).to_numpy()
        )
        model = lgb.LGBMRegressor(**settings)
        model.fit(
            x.loc[train_rows],
            y[train_rows],
            sample_weight=weights[train_rows],
            categorical_feature=list(CATEGORICAL),
        )
        prediction[future_rows] = model.predict(future.loc[future_rows])
        gain += model.booster_.feature_importance(importance_type="gain")
        fitted_groups.append(
            {
                "family_code": None if family is None else int(family),
                "training_rows": len(y[train_rows]),
                "forecast_rows": len(future.loc[future_rows]),
            }
        )
        if by_family and (index % 8 == 0 or index == len(groups)):
            print(f"Fitted {index}/{len(groups)} product families", flush=True)
        del model
    return prediction, gain, fitted_groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cutoff", help="Last date with observed sales, YYYY-MM-DD.")
    parser.add_argument(
        "--submission",
        action="store_true",
        help="Forecast official competition test IDs.",
    )
    parser.add_argument("--configuration", choices=CONFIGURATIONS, default="compact")
    parser.add_argument("--history-days", type=int, default=560)
    parser.add_argument("--origin-step", type=int, default=8)
    parser.add_argument("--max-origins", type=int, default=34)
    parser.add_argument("--half-life-days", type=float, default=0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if (
        min(args.history_days, args.origin_step, args.max_origins, args.threads) < 1
        or args.half_life_days < 0
    ):
        parser.error(
            "History, stride, origins, threads must be positive and half-life nonnegative."
        )
    if not args.submission and args.cutoff is None:
        parser.error("Supply --cutoff for a historical test or --submission.")
    # The optional learner is imported only when this experiment is executed.
    import lightgbm as lgb

    panel = load_data(args.data)
    cutoff_date = panel.train_end if args.submission else pd.Timestamp(args.cutoff)
    if cutoff_date > panel.train_end:
        parser.error("The forecast cutoff must be an observed training date.")
    if args.submission and args.cutoff and pd.Timestamp(args.cutoff) != panel.train_end:
        parser.error("--submission must use the last labeled date.")
    cutoff = panel.dates.get_loc(cutoff_date)
    targets = np.arange(cutoff + 1, cutoff + panel.horizon + 1)
    if targets[-1] >= len(panel.dates):
        parser.error("The panel does not contain the forecast horizon.")
    actual = panel.sales[:, targets]
    if not args.submission and not np.isfinite(actual).all():
        parser.error(
            "Historical scoring requires observed targets across the complete horizon."
        )

    stores, holidays, source_files = _load_calendar(args.data)
    flags = calendar_flags(panel, stores, holidays)
    store_features = _store_features(panel, stores)
    start = perf_counter()
    x, y, weights, future, origins = build_training(
        panel, cutoff, flags, store_features, args
    )
    feature_seconds = perf_counter() - start
    print(f"Training {len(y):,} rows x {len(x.columns)} features", flush=True)
    settings = {
        "objective": "regression",
        "max_bin": 127,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "random_state": 481,
        "deterministic": True,
        "force_col_wise": True,
        "histogram_pool_size": 128,
        "n_jobs": args.threads,
        "verbosity": -1,
        **CONFIGURATIONS[args.configuration],
    }
    start = perf_counter()
    log_prediction, gain, fitted_groups = fit_predictions(
        x, y, weights, future, settings, by_family=args.configuration == "family"
    )
    prediction = np.maximum(np.expm1(log_prediction), 0).reshape(
        len(panel.series), panel.horizon
    )
    model_seconds = perf_counter() - start
    metadata = {
        "configuration": args.configuration,
        "grouping": "family" if args.configuration == "family" else "global",
        "fitted_groups": fitted_groups,
        "settings": settings,
        "data": panel.provenance,
        "calendar_source_files": source_files,
        "cutoff": cutoff_date.date().isoformat(),
        "forecast_start": panel.dates[targets[0]].date().isoformat(),
        "forecast_end": panel.dates[targets[-1]].date().isoformat(),
        "origin_step": args.origin_step,
        "history_days": args.history_days,
        "feature_max_calendar_lookback": LOOKBACK,
        "training_origins": [panel.dates[o].date().isoformat() for o in origins],
        "training_rows": len(y),
        "features": list(x.columns),
        "categorical_features": CATEGORICAL,
        "half_life_days": args.half_life_days,
        "feature_seconds": feature_seconds,
        "model_seconds": model_seconds,
        "runtime": {
            "python": platform.python_version(),
            "lightgbm": lgb.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    if not args.submission:
        metadata["scores"] = _metrics(actual, prediction)
        print(json.dumps(metadata["scores"]), flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "prediction.npz",
        predictions=prediction,
        metadata=np.asarray(json.dumps(metadata)),
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(
        {
            "feature": x.columns,
            "gain": gain,
        }
    ).sort_values("gain", ascending=False).to_csv(
        args.output_dir / "feature-importance.csv", index=False
    )
    if args.submission:
        submission_frame(panel, panel.dates[targets], prediction).to_csv(
            args.output_dir / "submission.csv", index=False
        )


if __name__ == "__main__":
    main()
