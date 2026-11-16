import numpy as np
import pandas as pd
import pytest

from auxiliary.data import chronological_folds, make_fixture
from auxiliary.run import _forecast, _submission, errors


def test_kaggle_error_in_original_units():
    actual = np.array([[0.0, 3.0]])
    prediction = np.array([[1.0, 7.0]])

    log_error, absolute_error = errors(actual, prediction)

    np.testing.assert_allclose(log_error, np.log(2) ** 2)
    np.testing.assert_array_equal(absolute_error, [[1, 4]])
    with pytest.raises(ValueError, match="nonnegative"):
        errors(actual, -prediction)


def test_fold_targets_cannot_influence_seasonal_forecast():
    panel = make_fixture()
    fold = chronological_folds(panel)[0]
    forecast = _forecast(
        panel, "seasonal_naive", fold.train_end, fold.forecast_dates, None, 1
    )
    panel.sales[:, panel.dates > fold.train_end] = 10_000

    np.testing.assert_array_equal(
        _forecast(
            panel, "seasonal_naive", fold.train_end, fold.forecast_dates, None, 1
        ),
        forecast,
    )
    np.testing.assert_array_equal(forecast[:, 0], forecast[:, 7])


def test_windows_are_disjoint_and_precede_competition_test():
    panel = make_fixture()
    folds = chronological_folds(panel)

    assert [fold.name for fold in folds] == ["validation_1", "validation_2", "holdout"]
    for earlier, later in zip(folds, folds[1:]):
        assert earlier.forecast_dates[-1] == later.train_end
        assert len(earlier.forecast_dates.intersection(later.forecast_dates)) == 0
    assert folds[-1].forecast_dates[-1] == panel.train_end
    assert panel.test["date"].min() == panel.train_end + pd.Timedelta(days=1)


def test_submission_aligns_store_family_dates_to_test_id():
    panel = make_fixture()
    output = _submission(panel, "seasonal_naive", None, 1)

    assert list(output) == ["id", "sales"]
    assert len(output) == len(panel.series) * panel.horizon
    np.testing.assert_array_equal(output["id"], panel.test["id"])
    assert output["sales"].notna().all()
    assert (output["sales"] >= 0).all()
    assert (
        output.loc[
            panel.test["family"].eq("DRY") & panel.test["store_nbr"].eq(1), "sales"
        ]
        .eq(0)
        .all()
    )
