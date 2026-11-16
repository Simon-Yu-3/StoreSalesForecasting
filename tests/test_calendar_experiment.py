"""Protect holiday jurisdiction, transfers, and forecast-date alignment."""

import numpy as np
import pandas as pd

from auxiliary.calendar_experiment import (
    HOLIDAY_COLUMNS,
    calendar_flags,
    calendar_inputs,
)
from auxiliary.data import make_fixture
from auxiliary.final_experiment import (
    FUTURE_PROMOTION_MODEL,
    _hgb_inputs,
    protocol_folds,
)


def calendar_fixture():
    stores = pd.DataFrame(
        {
            "store_nbr": [4, 2, 1, 3],
            "city": ["Delta", "Beta", "Alpha", "Gamma"],
            "state": ["South", "North", "North", "South"],
        }
    )
    holidays = pd.DataFrame(
        [
            ("2016-07-01", "Holiday", "Local", "Alpha", True),
            ("2016-07-02", "Transfer", "Local", "Alpha", False),
            ("2016-07-03", "Holiday", "Regional", "North", False),
            ("2016-07-04", "Event", "National", "Ecuador", False),
            ("2016-07-05", "Bridge", "National", "Ecuador", False),
            ("2016-07-06", "Work Day", "National", "Ecuador", False),
            ("2016-07-07", "Additional", "National", "Ecuador", False),
        ],
        columns=("date", "type", "locale", "locale_name", "transferred"),
    )
    holidays["date"] = pd.to_datetime(holidays["date"])
    return stores, holidays


def test_holidays_use_effective_dates_and_affected_store_locations():
    panel = make_fixture()
    stores, holidays = calendar_fixture()
    flags = calendar_flags(panel, stores, holidays)
    first = panel.dates.get_loc("2016-07-01")
    assert not any(flags[name][:, first].any() for name in HOLIDAY_COLUMNS)
    np.testing.assert_array_equal(flags["local_holiday"][:, first + 1], [1, 0, 0, 0])
    np.testing.assert_array_equal(flags["regional_holiday"][:, first + 2], [1, 1, 0, 0])
    assert not any(flags[name][:, first + 3].any() for name in HOLIDAY_COLUMNS)
    assert flags["holiday_bridge"][:, first + 4].all()
    assert flags["substitute_workday"][:, first + 5].all()
    assert flags["national_holiday"][:, first + 6].all()
    assert flags["public_sector_payday"][:, panel.dates.get_loc("2016-07-15")].all()
    assert flags["public_sector_payday"][:, panel.dates.get_loc("2016-07-31")].all()


def test_calendar_rows_match_the_existing_fixed_forecast_design():
    panel = make_fixture()
    stores, holidays = calendar_fixture()
    flags = calendar_flags(panel, stores, holidays)
    # Use a July forecast cutoff with a local transfer inside the horizon.
    origin = panel.dates.get_loc(protocol_folds(panel)[0][1].train_end)
    x, _, target = _hgb_inputs(panel, origin, FUTURE_PROMOTION_MODEL)
    calendar_train, calendar_target = calendar_inputs(panel, origin, flags)

    assert x.index.equals(calendar_train.index)
    assert target.index.equals(calendar_target.index)
    expected = target["store"].eq(0) & target["lead"].eq(2)
    assert expected.any()
    np.testing.assert_array_equal(calendar_target["local_holiday"], expected)
