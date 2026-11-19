"""Protect forecast alignment and chronological blend selection."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from auxiliary.competition_backtest import CUTOFFS, _forecast_dates, _save_cache
from auxiliary.competition_refinement import (
    _consistent_settings,
    _read_refinement_cache,
    _verify_submission_components,
    evaluate,
)
from auxiliary.data import make_fixture
from auxiliary.foundation import MODEL_ID, MODEL_REVISION
from auxiliary.run import submission_frame


def write_single_cache(path, panel, cutoff, values, *, learner):
    dates = _forecast_dates(panel, cutoff)
    metadata = {
        "data": panel.provenance,
        "cutoff": cutoff,
        "forecast_start": dates[0].date().isoformat(),
        "forecast_end": dates[-1].date().isoformat(),
        "settings": {"random_state": 481},
        "features": ["weekday"],
        "calendar_source_files": {},
        "configuration": "fixture",
        "history_days": 112,
        "origin_step": 8,
        "half_life_days": 0,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = (
        {"hgb_400": values, "hgb_700": values}
        if learner == "hgb"
        else {"predictions": values}
    )
    if learner == "hgb":
        metadata["settings"]["max_iter"] = 700
        metadata["stages"] = [400, 700]
    np.savez(path, **keys, metadata=np.asarray(json.dumps(metadata)))
    return metadata


def test_cache_rejects_forecast_dates_mismatched_to_the_cutoff(tmp_path):
    panel = make_fixture()
    cutoff = CUTOFFS[0]
    shape = (len(panel.series), panel.horizon)
    path = tmp_path / "prediction.npz"
    metadata = write_single_cache(path, panel, cutoff, np.zeros(shape), learner="lgb")
    metadata["forecast_start"] = cutoff
    np.savez(
        path, predictions=np.zeros(shape), metadata=np.asarray(json.dumps(metadata))
    )
    with pytest.raises(ValueError, match="forecast dates"):
        _read_refinement_cache(panel, path, cutoff, ["predictions"])


def test_later_targets_cannot_change_the_selected_blend(tmp_path):
    panel = make_fixture()
    previous = tmp_path / "previous"
    hgb_dir = tmp_path / "hgb"
    lgb_dir = tmp_path / "lgb"
    old_hgb, old_t0 = {}, {}
    for cutoff in CUTOFFS:
        actual = panel.sales[:, panel.dates.get_indexer(_forecast_dates(panel, cutoff))]
        old_hgb[f"{cutoff}_submitted"] = actual * 1.1
        old_hgb[f"{cutoff}_calendar"] = actual * 1.1
        old_t0[cutoff] = actual * 0.8
        write_single_cache(
            hgb_dir / cutoff / "hgb.npz", panel, cutoff, actual * 1.1, learner="hgb"
        )
        write_single_cache(
            lgb_dir / cutoff / "prediction.npz",
            panel,
            cutoff,
            actual * 0.9,
            learner="lgb",
        )
    _save_cache(panel, previous, "hgb", old_hgb, {})
    _save_cache(
        panel,
        previous,
        "t0",
        old_t0,
        {"checkpoint": MODEL_ID, "revision": MODEL_REVISION},
    )

    initial_output = tmp_path / "initial"
    initial = evaluate(panel, previous, hgb_dir, lgb_dir, initial_output)
    changed_sales = panel.sales.copy()
    changed_sales[:, panel.dates > pd.Timestamp(CUTOFFS[3])] = 1_000_000
    changed = replace(panel, sales=changed_sales)
    changed_output = tmp_path / "changed"
    repeated = evaluate(changed, previous, hgb_dir, lgb_dir, changed_output)

    assert initial["selected"] == repeated["selected"]
    assert initial["later_rmsle"] != repeated["later_rmsle"]
    assert (initial_output / "selection-grid.csv").read_bytes() == (
        changed_output / "selection-grid.csv"
    ).read_bytes()


def test_cached_training_origins_cannot_include_unobserved_labels(tmp_path):
    panel = make_fixture()
    cutoff = CUTOFFS[0]
    values = np.zeros((len(panel.series), panel.horizon))
    path = tmp_path / "prediction.npz"
    metadata = write_single_cache(path, panel, cutoff, values, learner="lgb")
    metadata["training_origins"] = [cutoff]
    np.savez(path, predictions=values, metadata=np.asarray(json.dumps(metadata)))
    with pytest.raises(ValueError, match="training labels after its cutoff"):
        _read_refinement_cache(panel, path, cutoff, ["predictions"])


def test_blend_rejects_csv_from_the_wrong_boosting_stage(tmp_path):
    panel = make_fixture()
    cutoff = panel.train_end.date().isoformat()
    shape = (len(panel.series), panel.horizon)
    hgb400 = np.full(shape, 2.0)
    hgb700 = np.full(shape, 3.0)
    lgb = np.full(shape, 4.0)
    hgb_path = tmp_path / "hgb" / "hgb_400.csv"
    lgb_path = tmp_path / "lgb" / "submission.csv"
    hgb_metadata = write_single_cache(
        hgb_path.parent / "hgb.npz", panel, cutoff, hgb400, learner="hgb"
    )
    lgb_metadata = write_single_cache(
        lgb_path.parent / "prediction.npz", panel, cutoff, lgb, learner="lgb"
    )
    np.savez(
        hgb_path.parent / "hgb.npz",
        hgb_400=hgb400,
        hgb_700=hgb700,
        metadata=np.asarray(json.dumps(hgb_metadata)),
    )
    settings = _consistent_settings(
        {
            "fixture": {
                "hgb": {"metadata": hgb_metadata},
                "lgb": {"metadata": lgb_metadata},
            }
        }
    )
    selection = {
        "selected": {"hgb_stage": 700},
        "base_model_settings": settings,
    }
    dates = _forecast_dates(panel, cutoff)
    submission_frame(panel, dates, hgb400).to_csv(hgb_path, index=False)
    submission_frame(panel, dates, lgb).to_csv(lgb_path, index=False)
    with pytest.raises(ValueError, match="selected component forecast"):
        _verify_submission_components(panel, selection, hgb_path, lgb_path)

    submission_frame(panel, dates, hgb700).to_csv(hgb_path, index=False)
    hgb_values, lgb_values, _ = _verify_submission_components(
        panel, selection, hgb_path, lgb_path
    )
    np.testing.assert_array_equal(hgb_values, np.full(len(panel.test), 3.0))
    np.testing.assert_array_equal(lgb_values, np.full(len(panel.test), 4.0))

    selection["base_model_settings"]["lgb"]["configuration"] = "other_specification"
    with pytest.raises(ValueError, match="settings must match"):
        _verify_submission_components(panel, selection, hgb_path, lgb_path)
