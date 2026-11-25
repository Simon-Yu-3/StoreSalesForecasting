"""Check the example's calendar and forecast-information boundaries."""

from io import StringIO

import numpy as np
import pandas as pd
import pytest

from examples.programming_sample import (
    build_features,
    forecast,
    read_sales,
    synthetic_csv,
)


def test_sales_lag_preserves_a_missing_calendar_date():
    data = read_sales(synthetic_csv())
    features = build_features(data)
    series = features.loc[features["store_nbr"].eq(1) & features["family"].eq("BREAD")]
    assert series.loc[series["date"].eq("2025-01-22"), "log_lag_28"].isna().all()
    actual = data.loc[
        data["date"].eq("2024-12-26")
        & data["store_nbr"].eq(1)
        & data["family"].eq("BREAD"),
        "sales",
    ].iloc[0]
    observed = series.loc[series["date"].eq("2025-01-23"), "log_lag_28"].iloc[0]
    assert observed == np.log1p(actual)


def test_holdout_sales_cannot_change_any_of_the_16_forecasts():
    data = read_sales(synthetic_csv())
    initial, cutoff, _ = forecast(data)
    altered = data.copy()
    altered.loc[altered["date"] > cutoff, "sales"] = 1_000_000
    repeated, repeated_cutoff, _ = forecast(altered)
    assert repeated_cutoff == cutoff
    assert len(initial) == 6 * 16
    assert initial["date"].min() == cutoff + pd.DateOffset(days=1)
    assert initial["date"].max() == cutoff + pd.DateOffset(days=16)
    np.testing.assert_array_equal(initial["forecast"], repeated["forecast"])


@pytest.mark.parametrize(
    "invalid", ["missing_column", "negative", "duplicate", "nonnumeric"]
)
def test_csv_validation_rejects_invalid_records(invalid):
    data = read_sales(synthetic_csv()).iloc[:2].copy()
    if invalid == "missing_column":
        data = data.drop(columns="onpromotion")
    elif invalid == "negative":
        data.loc[0, "sales"] = -1
    elif invalid == "duplicate":
        data = pd.concat([data, data.iloc[:1]])
    else:
        data["sales"] = "invalid"
    with pytest.raises(ValueError):
        read_sales(StringIO(data.to_csv(index=False)))
