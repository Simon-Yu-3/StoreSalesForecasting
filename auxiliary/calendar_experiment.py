"""Compare fixed boosting models with and without forecast-date calendar information."""

import argparse
import json
import os
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from auxiliary.data import _open_data, _read_csv, load_data
from auxiliary.final_experiment import (
    FUTURE_PROMOTION_MODEL,
    PROTOCOL_VERSION,
    _fold_record,
    _hgb_forecast,
    _hgb_inputs,
    _sha256,
    protocol_folds,
)
from auxiliary.global_benchmark import _metrics, training_origins
from auxiliary.run import _metrics as _slice_metrics
from auxiliary.run import _write_csv, errors

HOLIDAY_TYPES = {"Holiday", "Transfer", "Additional"}
HOLIDAY_COLUMNS = (
    "national_holiday",
    "regional_holiday",
    "local_holiday",
    "holiday_bridge",
    "substitute_workday",
)
PAYDAY_COLUMNS = ("public_sector_payday",)
CALENDAR_MODELS = {
    FUTURE_PROMOTION_MODEL: (),
    f"{FUTURE_PROMOTION_MODEL}_holidays": HOLIDAY_COLUMNS,
    f"{FUTURE_PROMOTION_MODEL}_payday": PAYDAY_COLUMNS,
    f"{FUTURE_PROMOTION_MODEL}_calendar": (*HOLIDAY_COLUMNS, *PAYDAY_COLUMNS),
}


def _load_calendar(path):
    with _open_data(path) as open_file:
        stores, stores_source = _read_csv(
            open_file,
            "stores.csv",
            ("store_nbr", "city", "state", "type", "cluster"),
            {"store_nbr": "int64", "cluster": "int64"},
        )
        holidays, holiday_source = _read_csv(
            open_file,
            "holidays_events.csv",
            ("date", "type", "locale", "locale_name", "description", "transferred"),
            {"transferred": "bool"},
            dates=True,
        )
    return (
        stores,
        holidays,
        {"stores.csv": stores_source, "holidays_events.csv": holiday_source},
    )


def calendar_flags(panel, stores, holidays):
    """Return binary flags indexed by panel store position and calendar day."""
    store_ids = panel.series.levels[0]
    if not stores["store_nbr"].is_unique or set(stores["store_nbr"]) != set(store_ids):
        raise ValueError(
            "Calendar requires exactly one location record for each panel store."
        )
    stores = stores.set_index("store_nbr").loc[store_ids]
    shape = (len(store_ids), len(panel.dates))
    flags = {name: np.zeros(shape, dtype=bool) for name in HOLIDAY_COLUMNS}
    scopes = {
        "National": (np.ones(len(stores), dtype=bool), "national_holiday"),
        "Regional": (stores["state"], "regional_holiday"),
        "Local": (stores["city"], "local_holiday"),
    }
    schedule = holidays.loc[
        ~holidays["transferred"]
        & holidays["type"].isin(HOLIDAY_TYPES | {"Bridge", "Work Day"})
    ]
    for row in schedule.itertuples(index=False):
        date_index = panel.dates.get_indexer([row.date])[0]
        if date_index < 0:
            continue
        locations, scope = scopes[row.locale]
        affected = (
            locations if row.locale == "National" else locations.eq(row.locale_name)
        )
        flag = {
            "Bridge": "holiday_bridge",
            "Work Day": "substitute_workday",
        }.get(row.type, scope)
        flags[flag][affected, date_index] = True

    paydays = (panel.dates.day == 15) | panel.dates.is_month_end
    flags["public_sector_payday"] = np.broadcast_to(paydays, shape)
    return flags


def _target_flags(panel, origin, flags):
    targets = origin + np.arange(1, panel.horizon + 1)
    stores = np.asarray(panel.series.codes[0])
    return pd.DataFrame(
        {
            name: values[stores[:, None], targets].ravel()
            for name, values in flags.items()
        }
    )


def calendar_inputs(panel, cutoff, flags):
    """Align calendar rows to the fixed model's simulated forecast origins."""
    frames = []
    for origin in training_origins(panel, cutoff):
        labeled = np.isfinite(
            panel.sales[:, origin + 1 : origin + panel.horizon + 1]
        ).ravel()
        frames.append(_target_flags(panel, origin, flags).loc[labeled])
    return pd.concat(frames, ignore_index=True), _target_flags(panel, cutoff, flags)


def _slice_scores(fold_record, name, actual, predicted, future):
    log_error, absolute_error = errors(actual, predicted)
    affected = (
        future[list(HOLIDAY_COLUMNS)].any(axis=1).to_numpy().reshape(actual.shape)
    )
    payday = future[PAYDAY_COLUMNS[0]].to_numpy().reshape(actual.shape)
    for dimension, groups in (
        ("holiday", (("affected", affected), ("unaffected", ~affected))),
        ("payday", (("payday", payday), ("other", ~payday))),
    ):
        for group, mask in groups:
            if mask.any():
                yield {
                    "fold": fold_record["fold"],
                    "role": fold_record["role"],
                    "model": name,
                    "dimension": dimension,
                    "group": group,
                    **_slice_metrics(log_error[mask], absolute_error[mask]),
                }


def run(data_path, selection_path, output_dir):
    import sklearn

    panel = load_data(data_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if (
        selection["protocol_version"] != PROTOCOL_VERSION
        or selection["data"] != panel.provenance
        or selection["selected_model"] != FUTURE_PROMOTION_MODEL
        or not selection["complete_model_set"]
    ):
        raise ValueError(
            "Calendar analysis requires the recorded official HGB selection."
        )
    reference = selection["selected_hgb_configuration"]
    settings = selection["hgb_configurations"][reference]
    stores, holidays, sources = _load_calendar(data_path)
    flags = calendar_flags(panel, stores, holidays)
    scores = []
    slices = []
    for role, fold in protocol_folds(panel):
        record = _fold_record(role, fold)
        cutoff = panel.dates.get_loc(fold.train_end)
        actual = panel.sales[:, panel.dates.get_indexer(fold.forecast_dates)]
        start = perf_counter()
        x, y, future = _hgb_inputs(panel, cutoff, FUTURE_PROMOTION_MODEL)
        reference_feature_seconds = perf_counter() - start
        start = perf_counter()
        calendar_train, calendar_future = calendar_inputs(panel, cutoff, flags)
        calendar_feature_seconds = perf_counter() - start
        if not (
            x.index.equals(calendar_train.index)
            and future.index.equals(calendar_future.index)
        ):
            raise RuntimeError(
                "Calendar rows do not align with the fixed forecast origins."
            )
        for name, features in CALENDAR_MODELS.items():
            start = perf_counter()
            inputs = (x, y, future)
            if features:
                columns = list(features)
                inputs = (
                    x.join(calendar_train[columns].astype("float32")),
                    y,
                    future.join(calendar_future[columns].astype("float32")),
                )
            prediction = _hgb_forecast(panel, inputs, settings)
            seconds = perf_counter() - start + reference_feature_seconds
            if features:
                seconds += calendar_feature_seconds
            metrics = _metrics(actual, prediction)
            scores.append({**record, "model": name, **metrics, "seconds": seconds})
            slices.extend(
                _slice_scores(record, name, actual, prediction, calendar_future)
            )
            print(f"{fold.name} {name}: RMSLE={metrics['rmsle']:.5f}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "scores.csv", scores)
    _write_csv(output_dir / "slices.csv", slices)
    artifact_hashes = {
        name: _sha256(output_dir / name) for name in ("scores.csv", "slices.csv")
    }
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "analysis": "Retrospective calendar ablation. The frozen model selection is unchanged.",
        "data": panel.provenance,
        "calendar_source_files": sources,
        "selection_manifest_sha256": _sha256(selection_path),
        "hgb_configuration": reference,
        "hgb_settings": settings,
        "folds": [_fold_record(role, fold) for role, fold in protocol_folds(panel)],
        "models": CALENDAR_MODELS,
        "assumption": (
            "The public calendar supplied with the competition approximates information "
            "available at the forecast origin. Announcement dates and contemporaneous "
            "schedule versions are unavailable."
        ),
        "payday_rule": "Public-sector salary schedule reported by Kaggle, the 15th and month end.",
        "artifact_sha256": artifact_hashes,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "omp_threads": os.environ.get("OMP_NUM_THREADS"),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.data, args.selection_manifest, args.output_dir)


if __name__ == "__main__":
    main()
