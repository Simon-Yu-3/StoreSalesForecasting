"""Exercise native tree learners and a keyed export on generated retail data."""

import importlib.util
import json
from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from auxiliary import provenance
from auxiliary.calendar_experiment import calendar_flags
from auxiliary.competition_ensemble import _with_calendar, _write_artifact
from auxiliary.competition_lightgbm import (
    _store_features,
    build_training,
    fit_predictions,
)
from auxiliary.competition_refinement import blend_log_sales
from auxiliary.data import make_fixture
from auxiliary.final_experiment import (
    FUTURE_PROMOTION_MODEL,
    _hgb_forecast,
    _hgb_inputs,
)
from auxiliary.global_benchmark import MODEL_SETTINGS
from auxiliary.run import _forecast, submission_frame


def test_source_provenance_changes_when_feature_implementation_changes(
    tmp_path, monkeypatch
):
    source = tmp_path / "features.py"
    monkeypatch.setattr(provenance, "SOURCE_DIRECTORY", tmp_path)
    source.write_text("version = 1\n", encoding="utf-8")
    first = provenance.feature_provenance("features.py")
    source.write_text("version = 2\n", encoding="utf-8")
    second = provenance.feature_provenance("features.py")
    assert first["feature_schema_version"] == second["feature_schema_version"] == 1
    assert first["source_file_sha256"] != second["source_file_sha256"]
    assert (
        second["source_file_sha256"]["auxiliary/features.py"]
        == sha256(source.read_bytes()).hexdigest()
    )


@pytest.mark.integration
def test_native_tree_forecasts_blend_and_export_generated_panel(tmp_path):
    if importlib.util.find_spec("lightgbm") is None:
        pytest.skip("Install the competition extra to run the native model test.")
    panel = make_fixture()
    test_bytes = panel.test.to_csv(index=False).encode("utf-8")
    panel = replace(
        panel,
        provenance={
            **panel.provenance,
            "files": {"test.csv": {"sha256": sha256(test_bytes).hexdigest()}},
        },
    )
    stores = pd.DataFrame(
        {
            "store_nbr": panel.series.levels[0],
            "city": ["North City", "North City", "South City", "South City"],
            "state": ["North", "North", "South", "South"],
            "type": ["A", "B", "A", "B"],
            "cluster": [1, 2, 1, 2],
        }
    )
    holidays = pd.DataFrame(
        {
            "date": pd.to_datetime(["2017-08-20"]),
            "type": ["Holiday"],
            "locale": ["Local"],
            "locale_name": ["North City"],
            "transferred": [False],
        }
    )
    flags = calendar_flags(panel, stores, holidays)
    cutoff = panel.dates.get_loc(panel.train_end)
    dates = panel.dates[cutoff + 1 : cutoff + panel.horizon + 1]
    args = SimpleNamespace(
        history_days=560, origin_step=8, max_origins=4, half_life_days=0
    )
    x, y, weights, future, origins = build_training(
        panel, cutoff, flags, _store_features(panel, stores), args
    )
    assert len(origins) == 4
    assert len(x.columns) == 70
    log_forecast, gain, groups = fit_predictions(
        x,
        y,
        weights,
        future,
        {
            "objective": "regression",
            "n_estimators": 8,
            "learning_rate": 0.1,
            "num_leaves": 7,
            "min_child_samples": 4,
            "n_jobs": 1,
            "random_state": 481,
            "deterministic": True,
            "force_col_wise": True,
            "verbosity": -1,
        },
        by_family=True,
    )
    lgb_forecast = np.maximum(np.expm1(log_forecast), 0).reshape(
        len(panel.series), panel.horizon
    )
    assert gain.sum() > 0
    assert len(groups) == len(panel.series.levels[1])
    assert sum(group["forecast_rows"] for group in groups) == len(panel.test)
    assert np.unique(lgb_forecast).size > 1

    hgb_inputs = _with_calendar(
        panel, cutoff, flags, _hgb_inputs(panel, cutoff, FUTURE_PROMOTION_MODEL)
    )
    hgb_forecast = _hgb_forecast(
        panel,
        hgb_inputs,
        {**MODEL_SETTINGS, "max_iter": 6, "max_leaf_nodes": 7, "min_samples_leaf": 4},
    )
    # A seasonal baseline supplies the third synthetic component.
    # The test exercises the blend interface without invoking a foundation model.
    seasonal_forecast = _forecast(
        panel, "seasonal_naive", panel.train_end, dates, None, jobs=1
    )
    forecast = blend_log_sales(
        hgb_forecast,
        lgb_forecast,
        seasonal_forecast,
        {"hgb_weight": 0.5, "lgb_weight": 0.25, "t0_weight": 0.25},
    )
    frame = submission_frame(panel, dates, forecast)
    manifest = _write_artifact(
        panel,
        tmp_path,
        "submission.csv",
        "manifest.json",
        frame,
        {
            "synthetic": True,
            "components": ["histogram_gradient_boosting", "lightgbm", "seasonal_naive"],
        },
    )
    exported = pd.read_csv(tmp_path / "submission.csv", float_precision="round_trip")
    pd.testing.assert_frame_equal(exported, frame, check_exact=True)
    np.testing.assert_array_equal(exported["id"], panel.test["id"])
    assert len(exported) == len(panel.series) * panel.horizon
    assert np.isfinite(exported["sales"]).all() and exported["sales"].ge(0).all()
    assert manifest == json.loads((tmp_path / "manifest.json").read_text())
    assert (
        manifest["sha256"]
        == sha256((tmp_path / "submission.csv").read_bytes()).hexdigest()
    )
    assert manifest["synthetic"]
