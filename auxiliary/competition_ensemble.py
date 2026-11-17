"""Reproduce the competition ensemble as independent model and blend stages."""

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from auxiliary.calendar_experiment import (
    _load_calendar,
    calendar_flags,
    calendar_inputs,
)
from auxiliary.data import load_data
from auxiliary.final_experiment import (
    FUTURE_PROMOTION_MODEL,
    PROTOCOL_VERSION,
    _hgb_forecast,
    _hgb_inputs,
    _sha256,
)
from auxiliary.run import _package_versions, _submission, submission_frame

CALENDAR_WEIGHT = 0.75
T0_WEIGHT = 0.25
T0_MODEL = "t0_beta_promotion"
COLUMNS = ["id", "sales"]


def _read_predictions(path, expected_ids):
    frame = pd.read_csv(
        path, dtype={"id": "int64", "sales": "float64"}, float_precision="round_trip"
    )
    if (
        list(frame.columns) != COLUMNS
        or not isinstance(frame.index, pd.RangeIndex)
        or not np.array_equal(frame["id"], expected_ids)
    ):
        raise ValueError(f"{path} must have id,sales columns and match test ID order.")
    values = frame["sales"].to_numpy()
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{path} must contain finite nonnegative forecasts.")
    return values


def _test_dates(panel):
    return panel.dates[panel.dates > panel.train_end]


def _write_artifact(panel, output_dir, filename, manifest_name, frame, details):
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / filename
    frame.to_csv(destination, index=False, lineterminator="\n")
    test_dates = _test_dates(panel)
    metadata = {
        **details,
        "filename": filename,
        "sha256": _sha256(destination),
        "rows": len(frame),
        "train_end": panel.train_end.date().isoformat(),
        "forecast_start": test_dates[0].date().isoformat(),
        "forecast_end": test_dates[-1].date().isoformat(),
        "test_sha256": panel.provenance["files"]["test.csv"]["sha256"],
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    (output_dir / manifest_name).write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def _hgb_settings(panel, selection_path):
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if (
        selection["protocol_version"] != PROTOCOL_VERSION
        or selection["data"] != panel.provenance
        or selection["selected_model"] != FUTURE_PROMOTION_MODEL
        or not selection["complete_model_set"]
    ):
        raise ValueError("Calendar HGB requires the recorded official HGB selection.")
    configuration = selection["selected_hgb_configuration"]
    return configuration, selection["hgb_configurations"][configuration]


def _with_calendar(panel, cutoff, flags, base_inputs):
    x, y, future = base_inputs
    calendar_train, calendar_future = calendar_inputs(panel, cutoff, flags)
    if not (
        x.index.equals(calendar_train.index)
        and future.index.equals(calendar_future.index)
    ):
        raise RuntimeError(
            "Calendar rows do not align with the fixed forecast origins."
        )
    return (
        x.join(calendar_train.astype("float32")),
        y,
        future.join(calendar_future.astype("float32")),
    )


def calendar_hgb(panel, data_path, selection_path, output_dir):
    """Fit the recorded HGB configuration with effective holidays and paydays."""
    import sklearn

    configuration, settings = _hgb_settings(panel, selection_path)
    cutoff = panel.dates.get_loc(panel.train_end)
    stores, holidays, sources = _load_calendar(data_path)
    flags = calendar_flags(panel, stores, holidays)
    base_inputs = _hgb_inputs(panel, cutoff, FUTURE_PROMOTION_MODEL)
    data = _with_calendar(panel, cutoff, flags, base_inputs)
    forecast = _hgb_forecast(panel, data, settings)
    frame = submission_frame(panel, _test_dates(panel), forecast)
    return _write_artifact(
        panel,
        output_dir,
        "calendar_hgb.csv",
        "calendar_hgb.json",
        frame,
        {
            "model": f"{FUTURE_PROMOTION_MODEL}_calendar",
            "configuration": configuration,
            "settings": settings,
            "scikit_learn": sklearn.__version__,
            "selection_manifest_sha256": _sha256(selection_path),
            "calendar_source_files": sources,
        },
    )


def t0_beta(panel, device, output_dir):
    """Forecast with the same promotion inputs and pinned t0-beta checkpoint."""
    from auxiliary.foundation import MODEL_ID, MODEL_REVISION, T0Forecaster

    forecaster = T0Forecaster(device=device)
    frame = _submission(panel, T0_MODEL, forecaster, jobs=1)
    return _write_artifact(
        panel,
        output_dir,
        "t0_beta.csv",
        "t0_beta.json",
        frame,
        {
            "model": T0_MODEL,
            "checkpoint": MODEL_ID,
            "revision": MODEL_REVISION,
            "device": device,
            "batch_size": forecaster.batch_size,
            "packages": _package_versions((T0_MODEL,), device),
        },
    )


def blend_sales(hgb_sales, t0_sales, t0_weight):
    return np.expm1(
        (1 - t0_weight) * np.log1p(hgb_sales) + t0_weight * np.log1p(t0_sales)
    )


def blend(panel, calendar_path, t0_path, output_dir):
    """Validate the official keyed rows and blend in the RMSLE target scale."""
    ids = panel.test["id"].to_numpy()
    calendar_prediction = _read_predictions(calendar_path, ids)
    t0_prediction = _read_predictions(t0_path, ids)
    prediction = blend_sales(calendar_prediction, t0_prediction, T0_WEIGHT)
    frame = pd.DataFrame({"id": ids, "sales": prediction})
    return _write_artifact(
        panel,
        output_dir,
        "submission.csv",
        "manifest.json",
        frame,
        {
            "calendar_hgb_weight": CALENDAR_WEIGHT,
            "t0_beta_weight": T0_WEIGHT,
            "blend_scale": "log1p sales",
            "inputs_sha256": {
                "calendar_hgb": _sha256(calendar_path),
                T0_MODEL: _sha256(t0_path),
            },
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    calendar = stages.add_parser(
        "calendar_hgb", help="Refit HGB with calendar features."
    )
    foundation = stages.add_parser(
        "t0_beta", help="Run the pinned promotion-aware t0-beta."
    )
    combine = stages.add_parser(
        "blend", help="Blend the two official prediction files."
    )
    for stage in (calendar, foundation, combine):
        stage.add_argument(
            "--data", type=Path, required=True, help="Official Kaggle ZIP or directory."
        )
        stage.add_argument("--output-dir", type=Path, required=True)
    calendar.add_argument(
        "--selection-manifest",
        type=Path,
        default=Path("files/final-results/manifest.json"),
    )
    foundation.add_argument(
        "--device", choices=("cpu", "mlx", "mps", "cuda"), default="cpu"
    )
    combine.add_argument("--calendar-csv", type=Path)
    combine.add_argument("--t0-csv", type=Path)
    args = parser.parse_args()

    panel = load_data(args.data)
    if args.stage == "calendar_hgb":
        calendar_hgb(panel, args.data, args.selection_manifest, args.output_dir)
    elif args.stage == "t0_beta":
        t0_beta(panel, args.device, args.output_dir)
    else:
        calendar_path = args.calendar_csv or args.output_dir / "calendar_hgb.csv"
        t0_path = args.t0_csv or args.output_dir / "t0_beta.csv"
        blend(panel, calendar_path, t0_path, args.output_dir)


if __name__ == "__main__":
    main()
