"""Describe the labeled sales panel using aggregates suitable for the report."""

import argparse
import json
from calendar import day_name
from hashlib import file_digest
from pathlib import Path

import numpy as np
import pandas as pd

from auxiliary.data import Panel, load_data


def _summarize_days(daily: pd.DataFrame, groups) -> pd.DataFrame:
    summary = daily.groupby(groups, sort=False).agg(
        days=("observations", "size"),
        observations=("observations", "sum"),
        total_sales=("total_sales", "sum"),
        zero_sales=("zero_sales", "sum"),
        promoted=("promoted", "sum"),
    )
    return summary.assign(
        mean_daily_sales=summary["total_sales"] / summary["days"],
        zero_sales_share=summary["zero_sales"] / summary["observations"],
        promoted_share=summary["promoted"] / summary["observations"],
    ).drop(columns=["total_sales", "zero_sales", "promoted"])


def profile(panel: Panel, output_dir: Path) -> None:
    """Write monthly and weekday summaries and their source hashes."""
    end = panel.dates.get_loc(panel.train_end) + 1
    sales = panel.sales[:, :end]
    promotions = panel.promotions[:, :end]
    observed = np.isfinite(sales)
    daily = pd.DataFrame(
        {
            "observations": observed.sum(axis=0),
            "total_sales": np.nansum(sales, axis=0),
            "zero_sales": (sales == 0).sum(axis=0),
            "promoted": (observed & (promotions > 0)).sum(axis=0),
        },
        index=panel.dates[:end],
    )
    daily = daily.loc[daily["observations"] > 0]

    months = daily.index.to_period("M").rename("month")
    monthly = _summarize_days(daily, months)
    complete = (monthly.index.start_time >= panel.dates[0]) & (
        monthly.index.end_time.normalize() <= panel.train_end
    )
    monthly.insert(0, "complete_month", complete.astype(int))

    weekday_summary = _summarize_days(daily, daily.index.dayofweek).sort_index()
    weekday_summary.index = weekday_summary.index.map(day_name.__getitem__).rename(
        "weekday"
    )

    recorded = sales[observed]
    total_rows = int(daily["observations"].sum())
    summary = {
        "observations": total_rows,
        "series": len(panel.series),
        "recorded_dates": len(daily),
        "first_date": daily.index[0].date().isoformat(),
        "last_date": daily.index[-1].date().isoformat(),
        "zero_sales_share": float(daily["zero_sales"].sum() / total_rows),
        "promoted_share": float(daily["promoted"].sum() / total_rows),
        "median_sales": float(np.median(recorded)),
        "mean_sales": float(recorded.mean()),
        "p90_sales": float(np.quantile(recorded, 0.9)),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for filename, frame in (("monthly.csv", monthly), ("weekday.csv", weekday_summary)):
        path = output_dir / filename
        frame.to_csv(path, lineterminator="\n")
        with path.open("rb") as artifact:
            hashes[filename] = file_digest(artifact, "sha256").hexdigest()

    (output_dir / "manifest.json").write_text(
        json.dumps(
            {"data": panel.provenance, "summary": summary, "artifact_sha256": hashes},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, required=True, help="Kaggle ZIP or folder."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    profile(load_data(args.data), args.output_dir)


if __name__ == "__main__":
    main()
