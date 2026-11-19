"""Compare boosting iteration counts from one calendar HGB fit at one origin."""

import argparse
import json
import os
import platform
from datetime import date
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from auxiliary.calendar_experiment import _load_calendar, calendar_flags
from auxiliary.competition_ensemble import _hgb_settings, _with_calendar
from auxiliary.data import load_data
from auxiliary.final_experiment import FUTURE_PROMOTION_MODEL, _hgb_inputs, _sha256
from auxiliary.global_benchmark import _metrics
from auxiliary.run import submission_frame

ITERATIONS = (150, 400, 700)


def _cutoff(value):
    try:
        return pd.Timestamp(date.fromisoformat(value))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Use a cutoff date in YYYY-MM-DD format."
        ) from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--cutoff", required=True, type=_cutoff)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-iter", type=int, default=700)
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        default=Path("files/final-results/manifest.json"),
    )
    parser.add_argument(
        "--submission",
        action="store_true",
        help="Forecast the official competition test.",
    )
    args = parser.parse_args()
    if args.max_iter < ITERATIONS[0]:
        parser.error("--max-iter must be at least 150 to produce a requested stage.")

    import sklearn
    from sklearn.ensemble import HistGradientBoostingRegressor

    panel = load_data(args.data)
    if args.cutoff not in panel.dates or args.cutoff > panel.train_end:
        parser.error("--cutoff must be a calendar date in the labeled training period.")
    if args.submission and args.cutoff != panel.train_end:
        parser.error(
            "--submission requires the last labeled training date as --cutoff."
        )
    cutoff = panel.dates.get_loc(args.cutoff)
    targets = np.arange(cutoff + 1, cutoff + panel.horizon + 1)
    if targets[-1] >= len(panel.dates):
        parser.error(
            "The panel must contain all forecast dates through cutoff + horizon."
        )
    actual = panel.sales[:, targets]
    if not args.submission and not np.isfinite(actual).all():
        parser.error(
            "Historical scoring requires observed sales for the complete horizon."
        )

    configuration, selected_settings = _hgb_settings(panel, args.selection_manifest)
    settings = {**selected_settings, "max_iter": args.max_iter}
    stores, holidays, source_files = _load_calendar(args.data)
    flags = calendar_flags(panel, stores, holidays)
    start = perf_counter()
    base = _hgb_inputs(panel, cutoff, FUTURE_PROMOTION_MODEL)
    x, y, future = _with_calendar(panel, cutoff, flags, base)
    feature_seconds = perf_counter() - start
    model = HistGradientBoostingRegressor(
        categorical_features=["store", "family"], **settings
    )
    start = perf_counter()
    model.fit(x, y)
    fit_seconds = perf_counter() - start
    requested = {stage for stage in ITERATIONS if stage <= args.max_iter}
    forecasts = {}
    scores = {}
    for stage, log_prediction in enumerate(model.staged_predict(future), start=1):
        if stage in requested:
            prediction = np.expm1(np.maximum(log_prediction, 0)).reshape(
                len(panel.series), panel.horizon
            )
            if not np.isfinite(prediction).all():
                raise RuntimeError(f"HGB stage {stage} returned nonfinite predictions.")
            name = f"hgb_{stage}"
            forecasts[name] = prediction
            if not args.submission:
                scores[name] = _metrics(actual, prediction)
        if stage == max(requested):
            break
    if len(forecasts) != len(requested):
        raise RuntimeError("The fitted model did not produce every requested stage.")

    metadata = {
        "model": "calendar_hgb_staged",
        "configuration": configuration,
        "settings": settings,
        "stages": sorted(requested),
        "data": panel.provenance,
        "calendar_source_files": source_files,
        "selection_manifest_sha256": _sha256(args.selection_manifest),
        "cutoff": args.cutoff.date().isoformat(),
        "train_end": args.cutoff.date().isoformat(),
        "forecast_start": panel.dates[targets[0]].date().isoformat(),
        "forecast_end": panel.dates[targets[-1]].date().isoformat(),
        "horizon": panel.horizon,
        "submission": args.submission,
        "training_rows": len(y),
        "features": list(x.columns),
        "categorical_features": ["store", "family"],
        "feature_seconds": feature_seconds,
        "fit_seconds": fit_seconds,
        "scores": scores,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "omp_threads": os.environ.get("OMP_NUM_THREADS"),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "hgb.npz",
        **forecasts,
        metadata=np.asarray(json.dumps(metadata)),
    )
    if args.submission:
        for name, forecast in forecasts.items():
            submission_frame(panel, panel.dates[targets], forecast).to_csv(
                args.output_dir / f"{name}.csv", index=False, lineterminator="\n"
            )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Calendar HGB through {args.cutoff.date()}: {len(y):,} training rows, "
        f"{fit_seconds:.2f}s fit, stages {', '.join(forecasts)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
