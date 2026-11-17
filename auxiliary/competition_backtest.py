"""Regenerate the six-window competition blend comparison and selection grid."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from auxiliary.calendar_experiment import _load_calendar, calendar_flags
from auxiliary.competition_ensemble import (
    T0_MODEL,
    _hgb_settings,
    _with_calendar,
    blend_sales,
)
from auxiliary.data import load_data
from auxiliary.final_experiment import (
    FUTURE_PROMOTION_MODEL,
    _hgb_forecast,
    _hgb_inputs,
    _sha256,
)
from auxiliary.foundation import MODEL_ID, MODEL_REVISION, T0Forecaster
from auxiliary.run import _forecast, errors

PERIODS = {
    "selection": ("2017-04-30", "2017-05-16", "2017-06-12"),
    "later_check": ("2017-06-28", "2017-07-14", "2017-07-30"),
}
CUTOFFS = (*PERIODS["selection"], *PERIODS["later_check"])
WEIGHTS = (0, 0.25, 0.5, 0.75, 1)
HGB_MODELS = ("submitted", "calendar")


def _forecast_dates(panel, cutoff):
    return pd.date_range(
        pd.Timestamp(cutoff) + pd.Timedelta(days=1), periods=panel.horizon
    )


def _save_cache(panel, output_dir, stage, forecasts, details):
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"stage": stage, "data": panel.provenance, "cutoffs": CUTOFFS, **details}
    np.savez_compressed(
        output_dir / f"{stage}.npz",
        **forecasts,
        metadata=np.asarray(json.dumps(metadata)),
    )


def _load_cache(panel, output_dir, stage):
    path = output_dir / f"{stage}.npz"
    with np.load(path, allow_pickle=False) as cache:
        metadata = json.loads(cache["metadata"].item())
        if (
            metadata["stage"] != stage
            or metadata["data"] != panel.provenance
            or tuple(metadata["cutoffs"]) != CUTOFFS
        ):
            raise ValueError(f"{path} does not match the official data and cutoffs.")
        if stage == "t0" and (metadata["checkpoint"], metadata["revision"]) != (
            MODEL_ID,
            MODEL_REVISION,
        ):
            raise ValueError(f"{path} does not match the pinned foundation checkpoint.")
        forecasts = {key: cache[key] for key in cache.files if key != "metadata"}
        expected_keys = (
            set(CUTOFFS)
            if stage == "t0"
            else {f"{cutoff}_{model}" for cutoff in CUTOFFS for model in HGB_MODELS}
        )
        expected_shape = (len(panel.series), panel.horizon)
        if set(forecasts) != expected_keys or any(
            values.shape != expected_shape
            or not np.isfinite(values).all()
            or (values < 0).any()
            for values in forecasts.values()
        ):
            raise ValueError(
                f"{path} must contain complete forecast matrices for every cutoff."
            )
        return forecasts


def hgb(panel, data_path, selection_path, output_dir):
    configuration, settings = _hgb_settings(panel, selection_path)
    stores, holidays, sources = _load_calendar(data_path)
    flags = calendar_flags(panel, stores, holidays)
    forecasts = {}
    for cutoff in CUTOFFS:
        index = panel.dates.get_loc(cutoff)
        base = _hgb_inputs(panel, index, FUTURE_PROMOTION_MODEL)
        forecasts[f"{cutoff}_submitted"] = _hgb_forecast(panel, base, settings)
        forecasts[f"{cutoff}_calendar"] = _hgb_forecast(
            panel, _with_calendar(panel, index, flags, base), settings
        )
        print(f"HGB forecasts through {cutoff}", flush=True)
    _save_cache(
        panel,
        output_dir,
        "hgb",
        forecasts,
        {
            "configuration": configuration,
            "selection_manifest_sha256": _sha256(selection_path),
            "calendar_source_files": sources,
        },
    )


def t0(panel, device, output_dir):
    forecaster = T0Forecaster(device=device)
    forecasts = {}
    for cutoff in CUTOFFS:
        forecasts[cutoff] = _forecast(
            panel,
            T0_MODEL,
            pd.Timestamp(cutoff),
            _forecast_dates(panel, cutoff),
            forecaster,
            jobs=1,
        )
        print(f"t0-beta forecasts through {cutoff}", flush=True)
    _save_cache(
        panel,
        output_dir,
        "t0",
        forecasts,
        {"checkpoint": MODEL_ID, "revision": MODEL_REVISION, "device": device},
    )


def _period_scores(panel, cutoffs, hgb_predictions, t0_predictions):
    by_window = {model: [] for model in HGB_MODELS}
    sample_sizes = []
    for cutoff in cutoffs:
        dates = _forecast_dates(panel, cutoff)
        actual = panel.sales[:, panel.dates.get_indexer(dates)]
        sample_sizes.append(actual.size)
        for model in HGB_MODELS:
            hgb_sales = hgb_predictions[f"{cutoff}_{model}"]
            t0_sales = t0_predictions[cutoff]
            scores = []
            for weight in WEIGHTS:
                forecast = blend_sales(hgb_sales, t0_sales, weight)
                log_error, _ = errors(actual, forecast)
                scores.append(float(np.sqrt(log_error.mean())))
            by_window[model].append(scores)

    pooled = {
        model: np.sqrt(
            np.average(np.square(scores), axis=0, weights=sample_sizes)
        ).tolist()
        for model, scores in by_window.items()
    }
    return by_window, pooled


def evaluate(panel, output_dir):
    hgb_predictions = _load_cache(panel, output_dir, "hgb")
    t0_predictions = _load_cache(panel, output_dir, "t0")
    selection, selection_pooled = _period_scores(
        panel, PERIODS["selection"], hgb_predictions, t0_predictions
    )
    weight_rows = [
        {"hgb_component": model, "t0_beta_weight": weight, "selection_rmsle": rmsle}
        for model in HGB_MODELS
        for weight, rmsle in zip(WEIGHTS, selection_pooled[model], strict=True)
    ]
    chosen = min(weight_rows, key=lambda row: row["selection_rmsle"])
    chosen_model, chosen_weight = chosen["hgb_component"], chosen["t0_beta_weight"]
    later, later_pooled = _period_scores(
        panel, PERIODS["later_check"], hgb_predictions, t0_predictions
    )

    rows = []
    for period, cutoffs in PERIODS.items():
        for cutoff in cutoffs:
            dates = _forecast_dates(panel, cutoff)
            actual = panel.sales[:, panel.dates.get_indexer(dates)]
            baseline = hgb_predictions[f"{cutoff}_submitted"]
            forecast = blend_sales(
                hgb_predictions[f"{cutoff}_{chosen_model}"],
                t0_predictions[cutoff],
                chosen_weight,
            )
            baseline_log, baseline_abs = errors(actual, baseline)
            forecast_log, forecast_abs = errors(actual, forecast)
            rows.append(
                {
                    "period": period,
                    "train_end": cutoff,
                    "forecast_start": dates[0].date().isoformat(),
                    "forecast_end": dates[-1].date().isoformat(),
                    "rows": actual.size,
                    "v1_rmsle": float(np.sqrt(baseline_log.mean())),
                    "v2_rmsle": float(np.sqrt(forecast_log.mean())),
                    "actual_sales_units": float(actual.sum()),
                    "v1_absolute_error_units": float(baseline_abs.sum()),
                    "v2_absolute_error_units": float(forecast_abs.sum()),
                }
            )

    report = {
        "weights_for_t0": WEIGHTS,
        "selection_cutoffs": PERIODS["selection"],
        "later_check_cutoffs": PERIODS["later_check"],
        "selected_calendar_arm": chosen_model,
        "selected_t0_weight": chosen_weight,
        "rmsle": {
            model: {
                "selection_by_window": selection[model],
                "later_by_window": later[model],
                "selection_pooled": selection_pooled[model],
                "later_pooled": later_pooled[model],
            }
            for model in HGB_MODELS
        },
    }
    pd.DataFrame(rows).to_csv(
        output_dir / "backtest.csv", index=False, lineterminator="\n"
    )
    pd.DataFrame(weight_rows).to_csv(
        output_dir / "blend-weight-scores.csv", index=False, lineterminator="\n"
    )
    (output_dir / "historical-blend-evaluation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Selected {chosen_model} HGB with t0-beta weight {chosen_weight}", flush=True
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    hgb_stage = stages.add_parser(
        "hgb", help="Forecast with the two HGB configurations."
    )
    t0_stage = stages.add_parser("t0", help="Forecast with the pinned t0-beta.")
    evaluate_stage = stages.add_parser(
        "evaluate", help="Score cached historical forecasts."
    )
    for stage in (hgb_stage, t0_stage, evaluate_stage):
        stage.add_argument(
            "--data", required=True, type=Path, help="Official Kaggle ZIP or directory."
        )
        stage.add_argument("--output-dir", required=True, type=Path)
    hgb_stage.add_argument(
        "--selection-manifest",
        type=Path,
        default=Path("files/final-results/manifest.json"),
    )
    t0_stage.add_argument(
        "--device", choices=("cpu", "mlx", "mps", "cuda"), default="cpu"
    )
    args = parser.parse_args()

    panel = load_data(args.data)
    if args.stage == "hgb":
        hgb(panel, args.data, args.selection_manifest, args.output_dir)
    elif args.stage == "t0":
        t0(panel, args.device, args.output_dir)
    else:
        evaluate(panel, args.output_dir)


if __name__ == "__main__":
    main()
