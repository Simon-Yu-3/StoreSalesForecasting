"""CSV-to-forecast example. The default demonstration uses synthetic sales."""

import argparse
from io import StringIO
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

KEYS = ["date", "store_nbr", "family"]
HORIZON = 16
FEATURES = ["store_nbr", "family", "weekday", "month", "onpromotion", "log_lag_28"]


def read_sales(source):
    """Read the sales schema and reject missing, duplicate, or invalid records."""
    data = pd.read_csv(
        source,
        usecols=KEYS + ["sales", "onpromotion"],
        dtype={
            "store_nbr": "int64",
            "family": "string",
            "sales": "float64",
            "onpromotion": "int64",
        },
    )
    data["date"] = pd.to_datetime(data["date"], format="%Y-%m-%d", errors="raise")
    numeric = data[["store_nbr", "sales", "onpromotion"]]
    if data.isna().any().any() or not np.isfinite(numeric).all().all():
        raise ValueError("Every input field must be present and numeric values finite.")
    if (
        data.duplicated(KEYS).any()
        or (numeric < 0).any().any()
        or (data["store_nbr"] == 0).any()
    ):
        raise ValueError("Duplicate keys, invalid store IDs, or negative counts.")
    return data.sort_values(KEYS).reset_index(drop=True)


def build_features(data):
    """A date join keeps lag 28 aligned across missing calendar dates."""
    lagged = data[KEYS + ["sales"]].rename(columns={"sales": "lag_28"})
    lagged["date"] = lagged["date"] + pd.DateOffset(days=28)
    features = data.merge(lagged, on=KEYS, how="left", validate="one_to_one")
    features["log_lag_28"] = np.log1p(features["lag_28"])
    features["weekday"] = features["date"].dt.dayofweek
    features["month"] = features["date"].dt.month
    categories = ["store_nbr", "family", "weekday", "month"]
    features[categories] = features[categories].astype("category")
    return features


def forecast(data):
    """Fit before one cutoff and forecast all 16 days from that cutoff."""
    cutoff = data["date"].max() - pd.DateOffset(days=HORIZON)
    recent = data.loc[data["date"] >= cutoff - pd.DateOffset(days=392)]
    features = build_features(recent)
    train = features.loc[
        features["date"].between(cutoff - pd.DateOffset(days=364), cutoff)
    ]
    test = features.loc[features["date"] > cutoff].copy()
    series = data[["store_nbr", "family"]].drop_duplicates()
    if (
        train.empty
        or len(test) != len(series) * HORIZON
        or test["date"].nunique() != HORIZON
    ):
        raise ValueError(
            "The holdout must contain every series on all 16 forecast dates."
        )
    model = HistGradientBoostingRegressor(
        max_iter=60,
        max_leaf_nodes=15,
        min_samples_leaf=15,
        categorical_features="from_dtype",
        random_state=481,
    )
    model.fit(train[FEATURES], np.log1p(train["sales"]))
    # Lag 28 is observed at the cutoff for every lead in a 16-day forecast.
    # The future promotion counts are assumed to be scheduled and known.
    test["forecast"] = np.maximum(np.expm1(model.predict(test[FEATURES])), 0)
    return test, cutoff, len(train)


def evaluate(data, output_dir, dataset):
    predicted, cutoff, training_rows = forecast(data)
    log_error = np.log1p(predicted["forecast"]) - np.log1p(predicted["sales"])
    summary = {
        "dataset": dataset,
        "train_end": cutoff.date().isoformat(),
        "forecast_start": predicted["date"].min().date().isoformat(),
        "forecast_end": predicted["date"].max().date().isoformat(),
        "training_rows": training_rows,
        "forecast_rows": len(predicted),
        "series": predicted.groupby(["store_nbr", "family"], observed=True).ngroups,
        "rmsle": float(np.sqrt(np.mean(log_error**2))),
        "mae": float(np.mean(np.abs(predicted["forecast"] - predicted["sales"]))),
        "plot_file": "forecast.png",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(output_dir / "metrics.csv", index=False)
    daily = predicted.groupby("date")[["sales", "forecast"]].sum()
    ax = daily.rename(columns={"sales": "Observed"}).plot(figsize=(8, 3), marker="o")
    ax.set(
        title=f"{dataset}: 16-day historical forecast",
        ylabel="Sales units",
        xlabel="Date",
    )
    ax.figure.tight_layout()
    ax.figure.savefig(output_dir / "forecast.png", dpi=160)
    plt.close(ax.figure)
    return summary


def synthetic_csv():
    """Make six artificial series, including one missing calendar date."""
    dates = pd.date_range("2024-11-01", periods=224)
    demo = pd.MultiIndex.from_product(
        [dates, [1, 2, 3], ["BREAD", "DAIRY"]], names=KEYS
    ).to_frame(index=False)
    rng = np.random.default_rng(481)
    demo["onpromotion"] = rng.integers(0, 3, len(demo))
    level = 8 + 5 * demo["store_nbr"] + 8 * demo["family"].eq("DAIRY")
    weekly = 1 + 0.25 * demo["date"].dt.dayofweek.ge(5)
    demo["sales"] = rng.poisson(level * weekly * (1 + 0.2 * demo["onpromotion"]))
    demo = demo.loc[demo["date"] != pd.Timestamp("2024-12-25")]
    return StringIO(demo.to_csv(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--fixture", action="store_true", help="Use synthetic data.")
    source.add_argument("--data", type=Path, help="Kaggle ZIP containing train.csv.")
    parser.add_argument("--output-dir", type=Path, default="results/programming-sample")
    args = parser.parse_args()
    if args.data:
        with ZipFile(args.data) as archive, archive.open("train.csv") as csv_file:
            data = read_sales(csv_file)
        dataset = "Official Kaggle training data"
    else:
        data, dataset = read_sales(synthetic_csv()), "Synthetic demonstration"
    print(evaluate(data, args.output_dir, dataset))


if __name__ == "__main__":
    main()
