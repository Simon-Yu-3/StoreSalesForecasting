"""Select a three-model log-sales blend on chronological cached forecasts."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from auxiliary.competition_backtest import (
    PERIODS,
    _forecast_dates,
    _load_cache,
)
from auxiliary.competition_ensemble import _read_predictions, _write_artifact
from auxiliary.data import load_data
from auxiliary.final_experiment import _sha256
from auxiliary.foundation import MODEL_ID, MODEL_REVISION
from auxiliary.global_benchmark import _metrics
from auxiliary.run import errors, submission_frame

HGB_STAGES = (400, 700)
LGB_WEIGHTS = (0, 0.25, 0.5, 0.75, 1)
T0_WEIGHTS = (0, 0.25, 0.5)
PROTOCOL_VERSION = 1


def candidate_grid():
    return [
        {
            "hgb_stage": stage,
            "hgb_weight": 1 - lgb - t0,
            "lgb_weight": lgb,
            "t0_weight": t0,
        }
        for stage in HGB_STAGES
        for lgb in LGB_WEIGHTS
        for t0 in T0_WEIGHTS
        if lgb + t0 <= 1
    ]


def _check_prediction(values, shape, path):
    if values.shape != shape or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(
            f"{path} must contain a finite nonnegative forecast for every row."
        )


def _read_refinement_cache(panel, path, cutoff, models, *, historical=True):
    with np.load(path, allow_pickle=False) as cached:
        metadata = json.loads(cached["metadata"].item())
        dates = _forecast_dates(panel, cutoff)
        if (
            metadata["data"] != panel.provenance
            or metadata["cutoff"] != cutoff
            or metadata.get("train_end", cutoff) != cutoff
            or metadata["forecast_start"] != dates[0].date().isoformat()
            or metadata["forecast_end"] != dates[-1].date().isoformat()
            or metadata.get("horizon", panel.horizon) != panel.horizon
        ):
            raise ValueError(f"{path} must match the data, cutoff, and forecast dates.")
        if historical and metadata.get("submission", False):
            raise ValueError(f"{path} must contain historical predictions.")
        if any(
            pd.Timestamp(origin) + pd.Timedelta(days=panel.horizon)
            > pd.Timestamp(cutoff)
            for origin in metadata.get("training_origins", [])
        ):
            raise ValueError(f"{path} contains training labels after its cutoff.")
        predictions = {}
        for model in models:
            values = cached[model].copy()
            _check_prediction(values, (len(panel.series), panel.horizon), path)
            predictions[model] = values
    return predictions, metadata


def _load_candidates(panel, hgb_dir, lgb_dir, cutoff):
    hgb_path = hgb_dir / cutoff / "hgb.npz"
    lgb_path = lgb_dir / cutoff / "prediction.npz"
    hgb, hgb_metadata = _read_refinement_cache(
        panel, hgb_path, cutoff, [f"hgb_{stage}" for stage in HGB_STAGES]
    )
    lgb, lgb_metadata = _read_refinement_cache(panel, lgb_path, cutoff, ["predictions"])
    if not set(HGB_STAGES).issubset(hgb_metadata["stages"]) or hgb_metadata["settings"][
        "max_iter"
    ] < max(HGB_STAGES):
        raise ValueError("The HGB cache must contain every declared boosting stage.")
    if hgb_metadata["calendar_source_files"] != lgb_metadata["calendar_source_files"]:
        raise ValueError(
            "The HGB and LightGBM caches must use the same calendar sources."
        )
    return (
        hgb,
        lgb["predictions"],
        {
            "hgb": {"sha256": _sha256(hgb_path), "metadata": hgb_metadata},
            "lgb": {"sha256": _sha256(lgb_path), "metadata": lgb_metadata},
        },
    )


def blend_log_sales(hgb, lgb, t0, recipe):
    weights = np.array(
        [recipe["hgb_weight"], recipe["lgb_weight"], recipe["t0_weight"]]
    )
    if (
        not np.isfinite(weights).all()
        or (weights < 0).any()
        or not np.isclose(weights.sum(), 1)
    ):
        raise ValueError("The three blend weights must be nonnegative and sum to one.")
    return np.expm1(
        weights[0] * np.log1p(hgb)
        + weights[1] * np.log1p(lgb)
        + weights[2] * np.log1p(t0)
    )


def _consistent_settings(records):
    """Each base learner must keep one specification across chronological windows."""
    expected = None
    for record in records.values():
        hgb = record["hgb"]["metadata"]
        lgb = record["lgb"]["metadata"]
        current = {
            "hgb": {
                key: hgb[key]
                for key in ("settings", "features", "calendar_source_files")
            },
            "lgb": {
                key: lgb[key]
                for key in (
                    "configuration",
                    "settings",
                    "features",
                    "calendar_source_files",
                    "history_days",
                    "origin_step",
                    "half_life_days",
                )
            },
        }
        current["lgb"].update(
            grouping=lgb.get("grouping", "global"),
            categorical_features=lgb.get("categorical_features", []),
            training_origin_count=len(lgb.get("training_origins", [])),
        )
        if expected is not None and current != expected:
            raise ValueError(
                "All cached windows must use consistent base-model settings and features."
            )
        expected = current
    return expected


def _verify_submission_components(panel, selection, hgb_path, lgb_path):
    """Bind the supplied CSVs to the selected model stage and specification."""
    chosen = selection["selected"]
    hgb_key = f"hgb_{chosen['hgb_stage']}"
    cutoff = panel.train_end.date().isoformat()
    hgb_artifact = hgb_path.parent / "hgb.npz"
    lgb_artifact = lgb_path.parent / "prediction.npz"
    hgb, hgb_metadata = _read_refinement_cache(
        panel, hgb_artifact, cutoff, [hgb_key], historical=False
    )
    lgb, lgb_metadata = _read_refinement_cache(
        panel, lgb_artifact, cutoff, ["predictions"], historical=False
    )
    current = _consistent_settings(
        {
            "submission": {
                "hgb": {"metadata": hgb_metadata},
                "lgb": {"metadata": lgb_metadata},
            }
        }
    )
    expected = selection["base_model_settings"]
    stage = chosen["hgb_stage"]
    if (
        stage not in hgb_metadata["stages"]
        or hgb_metadata["settings"]["max_iter"] < stage
    ):
        raise ValueError(
            "The HGB submission artifact must contain the selected iteration count."
        )
    # Earlier stages of the same fitted boosting path use the same predictors.
    current["hgb"]["settings"]["max_iter"] = expected["hgb"]["settings"]["max_iter"]
    if current != expected:
        raise ValueError(
            "Submission model settings must match the historical selection."
        )

    ids = panel.test["id"].to_numpy()
    dates = _forecast_dates(panel, cutoff)
    hgb_csv = _read_predictions(hgb_path, ids)
    lgb_csv = _read_predictions(lgb_path, ids)
    for path, matrix, values in (
        (hgb_path, hgb[hgb_key], hgb_csv),
        (lgb_path, lgb["predictions"], lgb_csv),
    ):
        expected_sales = submission_frame(panel, dates, matrix)["sales"].to_numpy()
        if not np.array_equal(values, expected_sales):
            raise ValueError(f"{path} does not match the selected component forecast.")
    return (
        hgb_csv,
        lgb_csv,
        {
            "hgb": _sha256(hgb_artifact),
            "lgb": _sha256(lgb_artifact),
        },
    )


def _verify_t0_submission(panel, path):
    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    dates = panel.dates[panel.dates > panel.train_end]
    if (
        metadata["checkpoint"] != MODEL_ID
        or metadata["revision"] != MODEL_REVISION
        or metadata["sha256"] != _sha256(path)
        or metadata["test_sha256"] != panel.provenance["files"]["test.csv"]["sha256"]
        or metadata["train_end"] != panel.train_end.date().isoformat()
        or metadata["forecast_start"] != dates[0].date().isoformat()
        or metadata["forecast_end"] != dates[-1].date().isoformat()
    ):
        raise ValueError(
            "The t0-beta CSV must match its checkpoint, test dates, and checksum record."
        )
    return _read_predictions(path, panel.test["id"].to_numpy())


def evaluate(panel, previous_cache, hgb_cache, lgb_cache, output_dir):
    old_hgb = _load_cache(panel, previous_cache, "hgb")
    t0 = _load_cache(panel, previous_cache, "t0")
    candidates = candidate_grid()
    squared_errors = np.zeros(len(candidates))
    n_rows = 0
    window_grid = []
    loaded = {}
    provenance = {}
    for cutoff in PERIODS["selection"]:
        hgb, lgb, provenance[cutoff] = _load_candidates(
            panel, hgb_cache, lgb_cache, cutoff
        )
        loaded[cutoff] = hgb, lgb
        actual = panel.sales[:, panel.dates.get_indexer(_forecast_dates(panel, cutoff))]
        for index, recipe in enumerate(candidates):
            prediction = blend_log_sales(
                hgb[f"hgb_{recipe['hgb_stage']}"], lgb, t0[cutoff], recipe
            )
            log_error, _ = errors(actual, prediction)
            squared_errors[index] += log_error.sum()
            window_grid.append(
                {
                    "cutoff": cutoff,
                    "candidate": index,
                    **recipe,
                    "n": actual.size,
                    "rmsle": float(np.sqrt(log_error.mean())),
                }
            )
        n_rows += actual.size

    pooled = np.sqrt(squared_errors / n_rows)
    chosen_index = int(np.argmin(pooled))
    chosen = {
        **candidates[chosen_index],
        "selection_rmsle": float(pooled[chosen_index]),
    }
    # The recipe is fixed before later-window targets are read or scored.
    for cutoff in PERIODS["later_check"]:
        hgb, lgb, provenance[cutoff] = _load_candidates(
            panel, hgb_cache, lgb_cache, cutoff
        )
        loaded[cutoff] = hgb, lgb
    settings = _consistent_settings(provenance)

    rows = []
    for period, cutoffs in PERIODS.items():
        for cutoff in cutoffs:
            dates = _forecast_dates(panel, cutoff)
            actual = panel.sales[:, panel.dates.get_indexer(dates)]
            hgb, lgb = loaded[cutoff]
            prediction = blend_log_sales(
                hgb[f"hgb_{chosen['hgb_stage']}"], lgb, t0[cutoff], chosen
            )
            previous = np.expm1(
                0.75 * np.log1p(old_hgb[f"{cutoff}_calendar"])
                + 0.25 * np.log1p(t0[cutoff])
            )
            row = {
                "period": period,
                "train_end": cutoff,
                "forecast_start": dates[0].date().isoformat(),
                "forecast_end": dates[-1].date().isoformat(),
                "actual_sales_units": float(actual.sum()),
            }
            for label, values in (("v2", previous), ("refinement", prediction)):
                row.update(
                    {
                        f"{label}_{key}": value
                        for key, value in _metrics(actual, values).items()
                    }
                )
            rows.append(row)

    later = [row for row in rows if row["period"] == "later_check"]
    later_rmsle = {
        label: float(
            np.sqrt(
                sum(r[f"{label}_n"] * r[f"{label}_rmsle"] ** 2 for r in later)
                / sum(r[f"{label}_n"] for r in later)
            )
        )
        for label in ("v2", "refinement")
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(window_grid).to_csv(
        output_dir / "selection-window-scores.csv", index=False
    )
    pd.DataFrame(
        [
            {"candidate": index, **recipe, "selection_rmsle": float(score)}
            for index, (recipe, score) in enumerate(
                zip(candidates, pooled, strict=True)
            )
        ]
    ).to_csv(output_dir / "selection-grid.csv", index=False)
    pd.DataFrame(rows).to_csv(output_dir / "backtest.csv", index=False)
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "data": panel.provenance,
        "periods": PERIODS,
        "selected": chosen,
        "selected_candidate": chosen_index,
        "later_rmsle": later_rmsle,
        "blend_scale": "log1p sales",
        "baseline": {"calendar_hgb_weight": 0.75, "t0_weight": 0.25},
        "grid": {
            "hgb_stages": HGB_STAGES,
            "lgb_weights": LGB_WEIGHTS,
            "t0_weights": T0_WEIGHTS,
        },
        "base_model_settings": settings,
        "cache_provenance": provenance,
        "previous_cache_sha256": {
            name: _sha256(previous_cache / f"{name}.npz") for name in ("hgb", "t0")
        },
        "artifact_sha256": {
            name: _sha256(output_dir / name)
            for name in (
                "selection-window-scores.csv",
                "selection-grid.csv",
                "backtest.csv",
            )
        },
        "interpretation": "Chronological selection with retrospective reuse of previously reviewed historical dates.",
        "kaggle_status": "unsubmitted",
        "kaggle_rmsle": None,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"selected": chosen, "later_rmsle": later_rmsle}), flush=True)
    return manifest


def blend(panel, selection_path, hgb_path, lgb_path, t0_path, output_dir):
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if (
        selection["protocol_version"] != PROTOCOL_VERSION
        or selection["data"] != panel.provenance
    ):
        raise ValueError(
            "The selection manifest must match the current protocol and dataset."
        )
    for artifact, digest in selection["artifact_sha256"].items():
        if _sha256(selection_path.parent / artifact) != digest:
            raise ValueError(f"{artifact} must match the selection manifest hash.")
    chosen = selection["selected"]
    if chosen["hgb_stage"] not in HGB_STAGES:
        raise ValueError("The selected HGB stage is outside this protocol.")
    ids = panel.test["id"].to_numpy()
    hgb_values, lgb_values, artifacts = _verify_submission_components(
        panel, selection, hgb_path, lgb_path
    )
    prediction = blend_log_sales(
        hgb_values,
        lgb_values,
        _verify_t0_submission(panel, t0_path),
        chosen,
    )
    return _write_artifact(
        panel,
        output_dir,
        "submission.csv",
        "manifest.json",
        pd.DataFrame({"id": ids, "sales": prediction}),
        {
            "model": "chronologically_selected_hgb_lightgbm_t0_blend",
            "kaggle_status": "unsubmitted",
            "kaggle_rmsle": None,
            "selected": chosen,
            "blend_scale": "log1p sales",
            "selection_manifest_sha256": _sha256(selection_path),
            "base_model_settings": selection["base_model_settings"],
            "model_artifact_sha256": artifacts,
            "inputs_sha256": {
                "hgb": _sha256(hgb_path),
                "lgb": _sha256(lgb_path),
                "t0": _sha256(t0_path),
            },
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    evaluation = stages.add_parser(
        "evaluate", help="Select and compare cached historical forecasts."
    )
    combine = stages.add_parser(
        "blend", help="Blend official test predictions using the selected recipe."
    )
    for stage in (evaluation, combine):
        stage.add_argument("--data", type=Path, required=True)
        stage.add_argument("--output-dir", type=Path, required=True)
    evaluation.add_argument("--previous-cache", type=Path, required=True)
    evaluation.add_argument("--hgb-cache", type=Path, required=True)
    evaluation.add_argument("--lgb-cache", type=Path, required=True)
    combine.add_argument("--selection-manifest", type=Path, required=True)
    combine.add_argument(
        "--hgb-csv",
        type=Path,
        required=True,
        help="Selected HGB stage CSV beside its hgb.npz artifact.",
    )
    combine.add_argument(
        "--lgb-csv",
        type=Path,
        required=True,
        help="LightGBM CSV beside its prediction.npz artifact.",
    )
    combine.add_argument(
        "--t0-csv",
        type=Path,
        required=True,
        help="t0-beta CSV with its matching .json record.",
    )
    args = parser.parse_args()
    panel = load_data(args.data)
    if args.stage == "evaluate":
        evaluate(
            panel, args.previous_cache, args.hgb_cache, args.lgb_cache, args.output_dir
        )
    else:
        blend(
            panel,
            args.selection_manifest,
            args.hgb_csv,
            args.lgb_csv,
            args.t0_csv,
            args.output_dir,
        )


if __name__ == "__main__":
    main()
