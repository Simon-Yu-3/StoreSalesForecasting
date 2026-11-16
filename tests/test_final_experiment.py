"""Guard temporal separation and the three promotion information sets."""

from dataclasses import replace
from hashlib import sha256

import numpy as np
import pandas as pd

from auxiliary.data import chronological_folds, make_fixture
from auxiliary.final_experiment import (
    HGB_GRID,
    _hgb_inputs,
    _pooled_rmsle,
    protocol_folds,
    run,
)


def test_predeclared_windows_are_disjoint_and_precede_original_benchmark():
    panel = make_fixture()
    folds = protocol_folds(panel)
    assert [role for role, _ in folds] == ["development"] * 6 + ["evaluation"] * 3
    all_targets = pd.DatetimeIndex([])
    for _, fold in folds:
        assert len(fold.forecast_dates) == panel.horizon
        assert fold.forecast_dates[0] == fold.train_end + pd.Timedelta(days=1)
        assert fold.forecast_dates.intersection(all_targets).empty
        all_targets = all_targets.union(fold.forecast_dates)
    assert all_targets.max() < chronological_folds(panel)[0].forecast_dates[0]
    assert all_targets.max() < panel.test["date"].min()


def test_only_future_promotion_variant_reads_the_forecast_schedule():
    panel = make_fixture()
    cutoff = panel.dates.get_loc(protocol_folds(panel)[0][1].train_end)
    changed_promotions = panel.promotions.copy()
    changed_promotions[:, cutoff + 1 :] += 100
    changed = replace(panel, promotions=changed_promotions)
    columns = {}
    for model in (
        "global_hgb_sales",
        "global_hgb_historical_promotion",
        "global_hgb_future_promotion",
    ):
        train, labels, forecast = _hgb_inputs(panel, cutoff, model)
        changed_train, changed_labels, changed_forecast = _hgb_inputs(
            changed, cutoff, model
        )
        pd.testing.assert_frame_equal(train, changed_train)
        np.testing.assert_array_equal(labels, changed_labels)
        columns[model] = set(forecast.columns)
        if model == "global_hgb_future_promotion":
            assert (
                forecast["planned_promotion"] < changed_forecast["planned_promotion"]
            ).all()
            forecast = forecast.drop(columns="planned_promotion")
            changed_forecast = changed_forecast.drop(columns="planned_promotion")
        pd.testing.assert_frame_equal(forecast, changed_forecast)
    assert columns["global_hgb_historical_promotion"] - columns["global_hgb_sales"] == {
        "promotion_mean_28"
    }
    assert columns["global_hgb_future_promotion"] - columns[
        "global_hgb_historical_promotion"
    ] == {"planned_promotion"}


def test_tuning_and_final_choice_are_reconstructible_from_development(
    tmp_path, monkeypatch
):
    panel = make_fixture()
    # Exercise the selection and reporting paths without fitting boosted trees.
    import auxiliary.final_experiment as experiment

    def lightweight_forecast(panel, inputs, settings):
        _, labels, _ = inputs
        value = np.expm1(labels.mean()) * settings["max_leaf_nodes"] / 43
        return np.full((len(panel.series), panel.horizon), value)

    monkeypatch.setattr(experiment, "_hgb_forecast", lightweight_forecast)
    manifest = run(
        panel, tmp_path, skip_foundation=False, cache_dir=None, device="cpu", jobs=1
    )
    tuning = pd.read_csv(tmp_path / "tuning.csv")
    scores = pd.read_csv(tmp_path / "scores.csv")
    development_names = {
        fold["fold"] for fold in manifest["folds"] if fold["role"] == "development"
    }

    assert set(tuning["fold"]) == development_names
    assert set(tuning["configuration"]) == set(HGB_GRID)
    assert len(tuning) == len(development_names) * len(HGB_GRID)
    tune_rmsle = _pooled_rmsle(tuning.to_dict("records"), "configuration")
    assert min(tune_rmsle, key=tune_rmsle.get) == manifest["selected_hgb_configuration"]

    development_scores = scores.loc[scores["role"].eq("development")]
    model_rmsle = _pooled_rmsle(development_scores.to_dict("records"), "model")
    assert min(model_rmsle, key=model_rmsle.get) == manifest["selected_model"]
    assert np.allclose(
        list(model_rmsle.values()), list(manifest["development_rmsle"].values())
    )
    for name in ("scores.csv", "slices.csv", "tuning.csv"):
        assert (
            manifest["artifact_sha256"][name]
            == sha256((tmp_path / name).read_bytes()).hexdigest()
        )
    assert manifest["synthetic"]
    assert not manifest["complete_model_set"]
