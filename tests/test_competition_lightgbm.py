"""Guard the origin boundary without importing the optional LightGBM learner."""

import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from auxiliary.competition_lightgbm import (
    _store_features,
    build_training,
    fit_predictions,
    training_origins,
)
from auxiliary.data import make_fixture


def inputs():
    panel = make_fixture()
    stores = pd.DataFrame(
        {
            "store_nbr": panel.series.levels[0],
            "city": ["A", "B", "C", "D"],
            "state": ["North", "North", "South", "South"],
            "type": ["A", "B", "A", "B"],
            "cluster": [1, 2, 1, 2],
        }
    )
    flags = {"national_holiday": np.zeros((len(stores), len(panel.dates)), dtype=bool)}
    args = SimpleNamespace(
        history_days=560, origin_step=8, max_origins=10, half_life_days=0
    )
    return panel, flags, _store_features(panel, stores), args


def test_training_and_forecasts_respect_the_complete_information_boundary():
    panel, flags, stores, args = inputs()
    cutoff = panel.dates.get_loc("2017-06-28")
    x, y, weights, future, origins = build_training(panel, cutoff, flags, stores, args)

    sales = panel.sales.copy()
    sales[:, cutoff + 1 :] = 1_000_000
    promotions = panel.promotions.copy()
    promotions[:, cutoff + panel.horizon + 1 :] = 1_000_000
    changed = replace(panel, sales=sales, promotions=promotions)
    changed_x, changed_y, changed_weights, changed_future, changed_origins = (
        build_training(changed, cutoff, flags, stores, args)
    )
    pd.testing.assert_frame_equal(x, changed_x, check_exact=True)
    pd.testing.assert_frame_equal(future, changed_future, check_exact=True)
    np.testing.assert_array_equal(y, changed_y)
    np.testing.assert_array_equal(weights, changed_weights)
    assert origins == changed_origins

    expected = []
    for origin in origins:
        target_positions = np.arange(origin + 1, origin + panel.horizon + 1)
        assert target_positions[-1] <= cutoff
        assert panel.dates[target_positions[-1]] <= panel.train_end
        assert np.all(np.diff(target_positions) == 1)
        labels = panel.sales[:, target_positions].ravel()
        expected.append(np.log1p(labels[np.isfinite(labels)]).astype(np.float32))
    np.testing.assert_array_equal(y, np.concatenate(expected))
    assert len(set(panel.dates[origins].dayofweek)) == 7


def test_only_horizon_promotion_features_respond_to_the_supplied_plan():
    panel, flags, stores, args = inputs()
    cutoff = panel.dates.get_loc("2017-06-28")
    x, y, _, future, _ = build_training(panel, cutoff, flags, stores, args)
    promotions = panel.promotions.copy()
    promotions[:, cutoff + 1 : cutoff + panel.horizon + 1] += 100
    changed = replace(panel, promotions=promotions)
    changed_x, changed_y, _, changed_future, _ = build_training(
        changed, cutoff, flags, stores, args
    )
    pd.testing.assert_frame_equal(x, changed_x, check_exact=True)
    np.testing.assert_array_equal(y, changed_y)
    expected = {
        "planned_promotion",
        "promotion_weekday_change",
        "promotion_level_change",
        "promotion_horizon_log_mean",
        "promotion_previous_plan",
        "promotion_next_plan",
    }
    differing = {
        column for column in future if not future[column].equals(changed_future[column])
    }
    assert differing == expected


def test_training_origins_reject_a_cutoff_after_the_last_label():
    panel, _, _, args = inputs()
    cutoff = panel.dates.get_loc(panel.train_end) + 1
    with pytest.raises(ValueError, match="observed official training date"):
        training_origins(
            panel,
            cutoff,
            history_days=args.history_days,
            origin_step=args.origin_step,
            max_origins=args.max_origins,
        )


def test_family_models_preserve_prediction_order_and_training_membership(monkeypatch):
    fitted = []

    class Regressor:
        def __init__(self, **settings):
            self.booster_ = SimpleNamespace(
                feature_importance=lambda importance_type: np.array([2.0, 3.0])
            )

        def fit(self, frame, labels, *, sample_weight, categorical_feature):
            assert len(frame["family"].unique()) == 1
            assert "family" in categorical_feature
            self.value = np.average(labels, weights=sample_weight)
            fitted.append(frame["family"].iloc[0])
            return self

        def predict(self, frame):
            return np.full(len(frame), self.value)

    monkeypatch.setitem(
        sys.modules, "lightgbm", SimpleNamespace(LGBMRegressor=Regressor)
    )
    frame = pd.DataFrame({"family": [1, 0, 1, 0], "value": [10, 20, 30, 40]})
    future = pd.DataFrame({"family": [1, 0, 0, 1], "value": [0, 0, 0, 0]})
    prediction, gain, groups = fit_predictions(
        frame, np.array([1, 2, 3, 4]), np.ones(4), future, {}, by_family=True
    )
    assert fitted == [0, 1]
    np.testing.assert_array_equal(prediction, [2, 3, 3, 2])
    np.testing.assert_array_equal(gain, [4, 6])
    assert [group["training_rows"] for group in groups] == [2, 2]
    assert [group["forecast_rows"] for group in groups] == [2, 2]
