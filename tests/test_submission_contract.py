"""Reject ambiguous forecasts before their values are attached to Kaggle IDs."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from auxiliary.data import make_fixture
from auxiliary.run import submission_frame


@pytest.mark.parametrize("extra_rows,extra_days", [(1, 0), (0, 1), (-1, 0), (0, -1)])
def test_submission_rejects_inexact_matrix_dimensions(extra_rows, extra_days):
    panel = make_fixture()
    dates = panel.dates[panel.dates > panel.train_end]
    prediction = np.zeros((len(panel.series) + extra_rows, panel.horizon + extra_days))
    with pytest.raises(ValueError, match="forecast shape"):
        submission_frame(panel, dates, prediction)


def test_submission_aligns_reordered_dates_and_test_rows():
    panel = make_fixture()
    test = panel.test.sample(frac=1, random_state=481).reset_index(drop=True)
    panel = replace(panel, test=test)
    dates = panel.dates[panel.dates > panel.train_end]
    prediction = (
        np.arange(len(panel.series))[:, None] * 100 + np.arange(panel.horizon)[None, :]
    ).astype(float)
    ordinary = submission_frame(panel, dates, prediction)
    reversed_dates = submission_frame(panel, dates[::-1], prediction[:, ::-1])
    pd.testing.assert_frame_equal(ordinary, reversed_dates)
    np.testing.assert_array_equal(ordinary["id"], test["id"])
    for row, value in zip(test.itertuples(), ordinary["sales"], strict=True):
        position = panel.series.get_loc((row.store_nbr, row.family))
        day = dates.get_loc(row.date)
        assert value == prediction[position, day]


def test_submission_rejects_repeated_forecast_dates():
    panel = make_fixture()
    dates = pd.DatetimeIndex([panel.train_end + pd.Timedelta(days=1)] * panel.horizon)
    with pytest.raises(ValueError, match="unique"):
        submission_frame(panel, dates, np.zeros((len(panel.series), panel.horizon)))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_submission_preserves_existing_prediction_precision(dtype):
    panel = make_fixture()
    dates = panel.dates[panel.dates > panel.train_end]
    prediction = np.full((len(panel.series), panel.horizon), 0.123456789, dtype=dtype)
    result = submission_frame(panel, dates, prediction)
    expected = pd.DataFrame(
        {
            "id": panel.test["id"],
            "sales": np.full(len(panel.test), prediction[0, 0], dtype=dtype),
        }
    )
    pd.testing.assert_frame_equal(result, expected)
    assert result.to_csv(index=False) == expected.to_csv(index=False)


@pytest.mark.parametrize("corruption", ["duplicate_id", "duplicate_key", "missing_row"])
def test_submission_rejects_incomplete_or_ambiguous_test_grid(corruption):
    panel = make_fixture()
    test = panel.test.copy()
    if corruption == "duplicate_id":
        test.loc[1, "id"] = test.loc[0, "id"]
    elif corruption == "duplicate_key":
        test.loc[1, ["date", "store_nbr", "family"]] = test.loc[
            0, ["date", "store_nbr", "family"]
        ]
    else:
        test = test.iloc[:-1]
    changed = replace(panel, test=test)
    dates = panel.dates[panel.dates > panel.train_end]
    with pytest.raises(ValueError, match="complete, unique"):
        submission_frame(changed, dates, np.zeros((len(panel.series), panel.horizon)))
