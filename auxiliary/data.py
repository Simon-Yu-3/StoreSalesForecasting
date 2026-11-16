"""Read the Favorita sales panel and construct fixed temporal backtests."""

from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import file_digest
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd

COMPETITION_URL = (
    "https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data"
)
SERIES_KEYS = ["store_nbr", "family"]
TEST_COLUMNS = ["id", "date", *SERIES_KEYS, "onpromotion"]
CSV_DTYPES = {
    "id": "int64",
    "store_nbr": "int64",
    "family": "category",
    "onpromotion": "int64",
}
TRAIN_START = pd.Timestamp("2013-01-01")
TRAIN_END = pd.Timestamp("2017-08-15")
TEST_END = pd.Timestamp("2017-08-31")
HORIZON = 16
TRAIN_ROWS = 3_000_888
TEST_ROWS = 28_512


@dataclass(frozen=True)
class Panel:
    dates: pd.DatetimeIndex
    series: pd.MultiIndex
    sales: np.ndarray
    promotions: np.ndarray
    test: pd.DataFrame
    train_end: pd.Timestamp
    horizon: int
    provenance: dict


@dataclass(frozen=True)
class Fold:
    name: str
    train_end: pd.Timestamp
    forecast_dates: pd.DatetimeIndex


@contextmanager
def _open_data(path: Path):
    if path.is_dir():
        yield lambda filename: (path / filename).open("rb")
        return

    try:
        archive = ZipFile(path)
    except BadZipFile as exc:
        raise ValueError(
            f"Expected the original Kaggle ZIP, got {path.name!r}."
        ) from exc

    with archive:
        yield archive.open


def _read_csv(open_file, filename, columns, dtypes, *, dates=False):
    try:
        with open_file(filename) as file:
            sha256 = file_digest(file, "sha256").hexdigest()
        with open_file(filename) as file:
            frame = pd.read_csv(file, dtype=dtypes)
    except (FileNotFoundError, KeyError) as exc:
        raise FileNotFoundError(
            f"Dataset must contain {filename!r} at its root."
        ) from exc

    if set(frame.columns) != set(columns):
        raise ValueError(f"Unexpected columns in {filename}: {frame.columns.tolist()}")
    if frame.isna().any().any():
        raise ValueError(f"Unexpected missing cell in {filename}.")
    if dates:
        frame["date"] = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="raise")
    return frame.loc[:, columns], {"rows": len(frame), "sha256": sha256}


def _validate_rows(frame, filename, *, rows, id_start):
    if len(frame) != rows or not np.array_equal(
        np.sort(frame["id"].to_numpy()), np.arange(id_start, id_start + rows)
    ):
        raise ValueError(f"Unexpected row count or ID coverage in {filename}.")
    if frame.duplicated(["date", *SERIES_KEYS]).any():
        raise ValueError(f"Duplicate date × store × family row in {filename}.")
    if (frame["onpromotion"] < 0).any():
        raise ValueError(f"Negative promotion count in {filename}.")


def _positions(frame, dates, stores, families):
    store_index = stores.get_indexer(frame["store_nbr"])
    family_index = families.get_indexer(frame["family"])
    date_index = dates.get_indexer(frame["date"])
    if min(store_index.min(), family_index.min(), date_index.min()) < 0:
        raise ValueError("Unexpected date, store number, or product family.")
    return store_index * len(families) + family_index, date_index


def load_data(path: Path | str) -> Panel:
    """Load original competition files from their ZIP or a flat CSV directory.

    A historical date that is absent in the CSV stays NaN for every series.
    The competition's 16 dates beyond the last labeled day have missing sales
    and observed promotion counts. File digests are recorded before parsing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No Kaggle archive or CSV directory at {path}. Download the data from {COMPETITION_URL}."
        )

    with _open_data(path) as open_file:
        train, train_source = _read_csv(
            open_file,
            "train.csv",
            [*TEST_COLUMNS, "sales"],
            {**CSV_DTYPES, "sales": "float64"},
            dates=True,
        )
        test, test_source = _read_csv(
            open_file,
            "test.csv",
            TEST_COLUMNS,
            CSV_DTYPES,
            dates=True,
        )
        sample, sample_source = _read_csv(
            open_file,
            "sample_submission.csv",
            ["id", "sales"],
            {"id": "int64", "sales": "float64"},
        )

    _validate_rows(train, "train.csv", rows=TRAIN_ROWS, id_start=0)
    _validate_rows(test, "test.csv", rows=TEST_ROWS, id_start=TRAIN_ROWS)
    test = test.sort_values("id", ignore_index=True)
    if not np.array_equal(sample["id"].to_numpy(), test["id"].to_numpy()):
        raise ValueError(
            "sample_submission.csv IDs must match sorted test.csv IDs exactly."
        )

    stores = pd.Index(train["store_nbr"].unique()).sort_values()
    families = pd.Index(train["family"].astype(str).unique()).sort_values()
    if len(stores) != 54 or len(families) != 33:
        raise ValueError("Expected all 54 stores and 33 product families.")
    series = pd.MultiIndex.from_product([stores, families], names=SERIES_KEYS)
    dates = pd.date_range(TRAIN_START, TEST_END, freq="D", name="date")

    observed = pd.DatetimeIndex(train["date"].unique()).sort_values()
    if (
        observed[0] != TRAIN_START
        or observed[-1] != TRAIN_END
        or len(observed) * len(series) != TRAIN_ROWS
    ):
        raise ValueError("Unexpected training dates or incomplete store-family series.")

    expected_test_dates = dates[dates > TRAIN_END]
    actual_test_dates = pd.DatetimeIndex(test["date"].unique()).sort_values()
    if (
        not actual_test_dates.equals(expected_test_dates)
        or len(expected_test_dates) != HORIZON
    ):
        raise ValueError("Expected the complete 16-day competition forecast window.")

    shape = (len(series), len(dates))
    sales = np.full(shape, np.nan, dtype=np.float64)
    promotions = np.full(shape, np.nan, dtype=np.float32)
    values = train["sales"].to_numpy()
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Observed sales must be finite and nonnegative.")
    train_positions = _positions(train, dates, stores, families)
    sales[train_positions] = values
    promotions[train_positions] = train["onpromotion"].to_numpy()
    promotions[_positions(test, dates, stores, families)] = test[
        "onpromotion"
    ].to_numpy()

    missing_dates = dates[dates <= TRAIN_END].difference(observed)
    provenance = {
        "source": "Kaggle Store Sales – Time Series Forecasting",
        "source_url": COMPETITION_URL,
        "synthetic": False,
        "files": {
            "train.csv": train_source,
            "test.csv": test_source,
            "sample_submission.csv": sample_source,
        },
        "missing_training_dates": missing_dates.strftime("%Y-%m-%d").tolist(),
    }
    return Panel(dates, series, sales, promotions, test, TRAIN_END, HORIZON, provenance)


def chronological_folds(panel: Panel) -> tuple[Fold, ...]:
    """Construct two validation windows and a later holdout with disjoint targets."""
    names = ("validation_1", "validation_2", "holdout")
    first_target = panel.train_end - pd.Timedelta(days=len(names) * panel.horizon - 1)
    folds = []
    for offset, name in enumerate(names):
        start = first_target + pd.Timedelta(days=offset * panel.horizon)
        train_end = start - pd.Timedelta(days=1)
        forecast_dates = pd.date_range(
            start, periods=panel.horizon, freq="D", name="date"
        )
        date_index = panel.dates.get_indexer(forecast_dates)
        if train_end not in panel.dates or (date_index < 0).any():
            raise ValueError(
                f"Panel does not contain the calendar dates needed for {name}."
            )
        if not np.isfinite(panel.sales[:, date_index]).all():
            raise ValueError(
                f"Expected observed targets for every series and date in {name}."
            )
        folds.append(Fold(name, train_end, forecast_dates))
    return tuple(folds)


def make_fixture() -> Panel:
    """Generate small synthetic data for checking pipeline mechanics."""
    rng = np.random.default_rng(481)
    dates = pd.date_range("2016-01-01", TEST_END, freq="D", name="date")
    series = pd.MultiIndex.from_product(
        [[1, 2, 3, 4], ["DRY", "FROZEN", "PRODUCE"]],
        names=SERIES_KEYS,
    )
    amplitude = np.geomspace(0.3, 120, len(series))[:, None]
    growth = np.linspace(0.75, 1.2, len(dates))
    weekly = np.array([0.85, 0.8, 0.9, 0.9, 1, 1.25, 1.15])[dates.dayofweek]
    promotions = rng.poisson(0.6, (len(series), len(dates))).astype(np.float32)
    sales = rng.poisson(amplitude * growth * weekly * (1 + 0.2 * promotions)).astype(
        np.float64
    )
    sales[0] = 0.0
    missing_date = pd.Timestamp("2016-12-25")
    sales[:, dates == missing_date] = np.nan
    promotions[:, dates == missing_date] = np.nan

    test_mask = dates > TRAIN_END
    sales[:, test_mask] = np.nan
    test = pd.MultiIndex.from_product(
        [dates[test_mask], *series.levels],
        names=["date", *SERIES_KEYS],
    ).to_frame(index=False)
    test["id"] = np.arange(len(test), dtype=np.int64)
    test["onpromotion"] = promotions[:, test_mask].T.ravel().astype(np.int64)
    provenance = {
        "source": "Synthetic smoke-test fixture",
        "synthetic": True,
        "random_seed": 481,
        "missing_training_dates": [missing_date.strftime("%Y-%m-%d")],
    }
    return Panel(
        dates,
        series,
        sales,
        promotions,
        test[TEST_COLUMNS],
        TRAIN_END,
        HORIZON,
        provenance,
    )
