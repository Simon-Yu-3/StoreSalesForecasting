"""Evaluate the frozen ECON 481 forecast protocol and generate a Kaggle submission."""

import argparse
import json
import os
import platform
from hashlib import file_digest
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from auxiliary.conventional import CONTEXT_LENGTH
from auxiliary.data import Fold, Panel, load_data, make_fixture
from auxiliary.global_benchmark import (
    MODEL_SETTINGS,
    ORIGIN_STEP,
    _metrics,
    fit_predict,
    origin_features,
    training_examples,
)
from auxiliary.run import (
    BASELINES,
    _forecast,
    _package_versions,
    _slice_scores,
    _write_csv,
    errors,
    submission_frame,
)

PROTOCOL_VERSION = 1
ORIGINS = {
    "development": (
        "2016-06-30",
        "2016-08-15",
        "2016-10-31",
        "2016-12-08",
        "2017-02-28",
        "2017-03-31",
    ),
    "evaluation": ("2017-04-30", "2017-05-16", "2017-06-12"),
}
SLOW_MODELS = ("prophet_promotion", "t0_beta_promotion")
HGB_VARIANTS = {
    "global_hgb_sales": (False, False),
    "global_hgb_historical_promotion": (True, False),
    "global_hgb_future_promotion": (True, True),
}
HGB_GRID = {
    "reference": MODEL_SETTINGS,
    "regularized": {**MODEL_SETTINGS, "max_leaf_nodes": 31, "l2_regularization": 3.0},
    "flexible": {**MODEL_SETTINGS, "max_leaf_nodes": 63, "l2_regularization": 0.5},
}
FUTURE_PROMOTION_MODEL = "global_hgb_future_promotion"


def protocol_folds(panel: Panel) -> tuple[tuple[str, Fold], ...]:
    """Build disjoint, fully observed 16-day windows at the declared origins."""
    folds = []
    prior_end = None
    for role, origins in ORIGINS.items():
        for offset, date in enumerate(origins, 1):
            origin = pd.Timestamp(date)
            forecast_dates = pd.date_range(
                origin + pd.Timedelta(days=1), periods=panel.horizon, name="date"
            )
            positions = panel.dates.get_indexer(forecast_dates)
            if prior_end is not None and forecast_dates[0] <= prior_end:
                raise ValueError("Forecast windows must be chronological and disjoint.")
            if (
                origin not in panel.dates
                or forecast_dates[-1] > panel.train_end
                or (positions < 0).any()
                or not np.isfinite(panel.sales[:, positions]).all()
            ):
                raise ValueError(f"Missing labeled dates in {role} at {date}.")
            fold = Fold(f"{role}_{offset}", origin, forecast_dates)
            folds.append((role, fold))
            prior_end = forecast_dates[-1]
    return tuple(folds)


def _fold_record(role, fold):
    return {
        "fold": fold.name,
        "role": role,
        "train_end": fold.train_end.date().isoformat(),
        "forecast_start": fold.forecast_dates[0].date().isoformat(),
        "forecast_end": fold.forecast_dates[-1].date().isoformat(),
    }


def _hgb_inputs(panel, cutoff, name):
    historical, future = HGB_VARIANTS[name]
    promotion_settings = {"historical_promotions": historical, "promotions": future}
    x, y = training_examples(panel, cutoff, **promotion_settings)
    target = origin_features(panel, cutoff, **promotion_settings)
    return x, y, target


def _hgb_forecast(panel, inputs, settings):
    return fit_predict(panel, *inputs, settings)


def _pooled_rmsle(rows, key):
    squared = {}
    counts = {}
    for row in rows:
        name = row[key]
        n = row["n"]
        squared[name] = squared.get(name, 0.0) + n * row["rmsle"] ** 2
        counts[name] = counts.get(name, 0) + n
    return {
        name: float(np.sqrt(total / counts[name])) for name, total in squared.items()
    }


def _tune_hgb(panel, development):
    score_rows = []
    forecasts = {}
    for _, fold in development:
        start = perf_counter()
        cutoff = panel.dates.get_loc(fold.train_end)
        inputs = _hgb_inputs(panel, cutoff, FUTURE_PROMOTION_MODEL)
        feature_seconds = perf_counter() - start
        actual = panel.sales[:, panel.dates.get_indexer(fold.forecast_dates)]
        for configuration, settings in HGB_GRID.items():
            start = perf_counter()
            predicted = _hgb_forecast(panel, inputs, settings)
            seconds = feature_seconds + perf_counter() - start
            forecasts[fold.name, configuration] = (predicted, seconds)
            squared_log_error, _ = errors(actual, predicted)
            rmsle = float(np.sqrt(squared_log_error.mean()))
            score_rows.append(
                {
                    "fold": fold.name,
                    "configuration": configuration,
                    "n": actual.size,
                    "rmsle": rmsle,
                }
            )
            print(f"Tuning {fold.name} {configuration}: RMSLE={rmsle:.5f}", flush=True)
    scores = _pooled_rmsle(score_rows, "configuration")
    chosen = min(scores, key=scores.get)
    chosen_forecasts = {
        fold.name: forecasts[fold.name, chosen] for _, fold in development
    }
    return score_rows, scores, chosen, chosen_forecasts


def _hgb_predictions(panel, folds, configuration, dev_forecasts):
    settings = HGB_GRID[configuration]
    forecasts = {}
    for role, fold in folds:
        cutoff = panel.dates.get_loc(fold.train_end)
        for name in HGB_VARIANTS:
            if name == FUTURE_PROMOTION_MODEL and role == "development":
                forecasts[fold.name, name] = dev_forecasts[fold.name]
                continue
            start = perf_counter()
            inputs = _hgb_inputs(panel, cutoff, name)
            prediction = _hgb_forecast(panel, inputs, settings)
            seconds = perf_counter() - start
            forecasts[fold.name, name] = prediction, seconds
            print(f"{name} {fold.name}: {seconds:.2f}s", flush=True)
    return forecasts


def _foundation(model, device):
    if model != "t0_beta_promotion":
        return None, {}
    from auxiliary.foundation import MODEL_ID, MODEL_REVISION, T0Forecaster

    start = perf_counter()
    foundation = T0Forecaster(device=device)
    return foundation, {
        "checkpoint": MODEL_ID,
        "revision": MODEL_REVISION,
        "device": device,
        "batch_size": foundation.batch_size,
        "load_seconds": perf_counter() - start,
        "torch_threads": os.environ.get("OMP_NUM_THREADS"),
    }


def _slow_predictions(panel, folds, model, device, jobs):
    foundation, metadata = _foundation(model, device)
    predictions = {}
    seconds = {}
    for _, fold in folds:
        start = perf_counter()
        predictions[fold.name] = _forecast(
            panel, model, fold.train_end, fold.forecast_dates, foundation, jobs
        )
        seconds[fold.name] = perf_counter() - start
        print(f"{model} {fold.name}: {seconds[fold.name]:.2f}s", flush=True)
    return predictions, {
        **metadata,
        "model": model,
        "data": panel.provenance,
        "origins": {fold.name: fold.train_end.date().isoformat() for _, fold in folds},
        "seconds": seconds,
        "python": platform.python_version(),
        "packages": _package_versions((model,), device),
        "jobs": jobs,
    }


def _read_slow_predictions(path, panel, folds, model):
    with np.load(path, allow_pickle=False) as cache:
        metadata = json.loads(cache["metadata"].item())
        origins = {fold.name: fold.train_end.date().isoformat() for _, fold in folds}
        if (
            metadata["data"] != panel.provenance
            or metadata["model"] != model
            or metadata["origins"] != origins
        ):
            raise ValueError(
                f"{path} does not match the data, model, and declared origins."
            )
        if model == "t0_beta_promotion":
            from auxiliary.foundation import MODEL_ID, MODEL_REVISION

            if (metadata["checkpoint"], metadata["revision"]) != (
                MODEL_ID,
                MODEL_REVISION,
            ):
                raise ValueError(
                    f"{path} does not match the pinned foundation checkpoint."
                )
        return {name: cache[name] for name in origins}, metadata


def _slice_results(panel, role, fold, name, actual, predicted):
    log_error, abs_error = errors(actual, predicted)
    indices = panel.dates.get_indexer(fold.forecast_dates)
    base = {"fold": fold.name, "role": role, "model": name}
    for row in _slice_scores(
        fold.name, name, log_error, abs_error, panel.promotions[:, indices]
    ):
        yield {**base, **row}

    cutoff = panel.dates.get_loc(fold.train_end)
    history = panel.sales[:, max(0, cutoff - 55) : cutoff + 1]
    counts = np.isfinite(history).sum(axis=1)
    rates = np.divide(
        (history == 0).sum(axis=1), counts, where=counts > 0, out=np.ones(len(counts))
    )
    for group, mask in (
        ("low", rates <= 0.1),
        ("middle", (rates > 0.1) & (rates <= 0.5)),
        ("high", rates > 0.5),
    ):
        if mask.any():
            group_log = log_error[mask]
            yield {
                **base,
                "dimension": "recent_zero_rate",
                "group": group,
                "n": group_log.size,
                "rmsle": float(np.sqrt(group_log.mean())),
                "mae": float(abs_error[mask].mean()),
            }


def _sha256(path):
    with path.open("rb") as file:
        return file_digest(file, "sha256").hexdigest()


def _artifact_hashes(output_dir):
    names = ("scores.csv", "slices.csv", "tuning.csv")
    return {name: _sha256(output_dir / name) for name in names}


def run(panel, output_dir, *, skip_foundation, cache_dir, device, jobs):
    """Tune and select on development only, then compare frozen configurations."""
    import sklearn

    folds = protocol_folds(panel)
    development = [(role, fold) for role, fold in folds if role == "development"]
    tune_rows, tune_scores, configuration, dev_forecasts = _tune_hgb(panel, development)
    hgb_forecasts = _hgb_predictions(panel, folds, configuration, dev_forecasts)
    synthetic = panel.provenance["synthetic"]
    slow_models = SLOW_MODELS[:-1] if skip_foundation else SLOW_MODELS
    if synthetic:
        slow_models = ()
    models = (*BASELINES, *slow_models, *HGB_VARIANTS)
    slow = {}
    for model in slow_models:
        if cache_dir is None:
            slow[model] = _slow_predictions(panel, folds, model, device, jobs)
        else:
            slow[model] = _read_slow_predictions(
                cache_dir / f"{model}.npz", panel, folds, model
            )

    scores, slices = [], []
    development_rmsle = None
    selected_model = None
    for role, fold in folds:
        if role == "evaluation" and selected_model is None:
            development_rmsle = _pooled_rmsle(scores, "model")
            selected_model = min(development_rmsle, key=development_rmsle.get)
        fold_record = _fold_record(role, fold)
        actual = panel.sales[:, panel.dates.get_indexer(fold.forecast_dates)]
        for name in models:
            start = perf_counter()
            if name in slow:
                cached_predictions, model_metadata = slow[name]
                predicted = cached_predictions[fold.name]
                seconds = model_metadata["seconds"][fold.name]
            elif name in HGB_VARIANTS:
                predicted, seconds = hgb_forecasts[fold.name, name]
            else:
                predicted = _forecast(
                    panel, name, fold.train_end, fold.forecast_dates, None, jobs
                )
                seconds = perf_counter() - start
            row = {
                **fold_record,
                "model": name,
                **_metrics(actual, predicted),
                "seconds": seconds,
            }
            scores.append(row)
            slices.extend(_slice_results(panel, role, fold, name, actual, predicted))
            print(f"{fold.name} {name}: RMSLE={row['rmsle']:.5f}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "scores.csv", scores)
    _write_csv(output_dir / "slices.csv", slices)
    _write_csv(output_dir / "tuning.csv", tune_rows)
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "synthetic": synthetic,
        "complete_model_set": not synthetic and slow_models == SLOW_MODELS,
        "data": panel.provenance,
        "artifact_sha256": _artifact_hashes(output_dir),
        "horizon": panel.horizon,
        "context_days": CONTEXT_LENGTH,
        "hgb_origin_stride_days": ORIGIN_STEP,
        "folds": [_fold_record(role, fold) for role, fold in folds],
        "models": list(models),
        "hgb_configurations": HGB_GRID,
        "hgb_tuning_rmsle": tune_scores,
        "hgb_tuning_model": FUTURE_PROMOTION_MODEL,
        "selected_hgb_configuration": configuration,
        "development_rmsle": development_rmsle,
        "selected_model": selected_model,
        "runtime": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "omp_threads": os.environ.get("OMP_NUM_THREADS"),
            "slow_models": {name: metadata for name, (_, metadata) in slow.items()},
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _write_submission(panel, manifest_path, output_dir, device, jobs):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["synthetic"] or not manifest["complete_model_set"]:
        raise ValueError(
            "The submission requires a completed official model comparison."
        )
    if (
        manifest["protocol_version"] != PROTOCOL_VERSION
        or manifest["data"] != panel.provenance
    ):
        raise ValueError(
            "Submission data and protocol must match the model-selection run."
        )

    dates = panel.dates[panel.dates > panel.train_end]
    model = manifest["selected_model"]
    cutoff = panel.dates.get_loc(panel.train_end)
    if model in HGB_VARIANTS:
        import sklearn

        configuration = manifest["selected_hgb_configuration"]
        inputs = _hgb_inputs(panel, cutoff, model)
        prediction = _hgb_forecast(
            panel, inputs, manifest["hgb_configurations"][configuration]
        )
        refit_metadata = {
            "configuration": configuration,
            "scikit_learn": sklearn.__version__,
        }
    elif model in (*BASELINES, *SLOW_MODELS):
        foundation, refit_metadata = _foundation(model, device)
        refit_metadata["packages"] = _package_versions((model,), device)
        prediction = _forecast(panel, model, panel.train_end, dates, foundation, jobs)
    else:
        raise ValueError(
            f"Unrecognized selection in model-comparison manifest: {model}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = output_dir / "submission.csv"
    submission = submission_frame(panel, dates, prediction)
    submission.to_csv(submission_path, index=False)
    record = {
        "selected_model": model,
        "development_rmsle": manifest["development_rmsle"][model],
        "selection_manifest_sha256": _sha256(manifest_path),
        "submission_sha256": _sha256(submission_path),
        "train_end": panel.train_end.date().isoformat(),
        "forecast_start": dates[0].date().isoformat(),
        "forecast_end": dates[-1].date().isoformat(),
        "rows": len(submission),
        "refit": {"python": platform.python_version(), **refit_metadata},
    }
    (output_dir / "submission-manifest.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--fixture", action="store_true", help="Use small synthetic data."
    )
    source.add_argument("--data", type=Path, help="Official Kaggle ZIP or directory.")
    standalone = parser.add_mutually_exclusive_group()
    standalone.add_argument(
        "--model-shard",
        choices=SLOW_MODELS,
        help="Only compute this model's local prediction cache.",
    )
    standalone.add_argument(
        "--submission-from",
        type=Path,
        metavar="MANIFEST",
        help="Refit the selected model and predict Kaggle test rows.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--prediction-cache", type=Path, help="Read local caches for the slow models."
    )
    parser.add_argument(
        "--device", choices=("cpu", "mlx", "mps", "cuda"), default="cpu"
    )
    parser.add_argument("--jobs", type=int, default=1, help="Concurrent Prophet fits.")
    parser.add_argument(
        "--skip-foundation",
        action="store_true",
        help="Skip t0 for a partial official benchmark.",
    )
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive.")
    if args.fixture and (
        args.model_shard or args.submission_from or args.prediction_cache
    ):
        parser.error(
            "Prediction caches and competition submission require official data."
        )
    if (args.model_shard or args.submission_from) and (
        args.prediction_cache or args.skip_foundation
    ):
        parser.error(
            "Model shards and submissions cannot use --prediction-cache or --skip-foundation."
        )

    panel = make_fixture() if args.fixture else load_data(args.data)
    if args.model_shard:
        predictions, metadata = _slow_predictions(
            panel, protocol_folds(panel), args.model_shard, args.device, args.jobs
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.output_dir / f"{args.model_shard}.npz",
            **predictions,
            metadata=np.asarray(json.dumps(metadata)),
        )
    elif args.submission_from:
        _write_submission(
            panel, args.submission_from, args.output_dir, args.device, args.jobs
        )
    else:
        run(
            panel,
            args.output_dir,
            skip_foundation=args.skip_foundation,
            cache_dir=args.prediction_cache,
            device=args.device,
            jobs=args.jobs,
        )


if __name__ == "__main__":
    main()
