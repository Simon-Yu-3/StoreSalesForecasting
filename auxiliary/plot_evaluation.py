"""Render a README figure from the archived forecast evaluation scores."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator, StrMethodFormatter

SALES = "global_hgb_sales"
PAST_PROMOTIONS = "global_hgb_historical_promotion"
SCHEDULE = "global_hgb_future_promotion"
FOUNDATION = "t0_beta_promotion"
ABLATION = [SALES, PAST_PROMOTIONS, SCHEDULE]
INK = "#172d30"
MUTED = "#4c6266"
RULE = "#dce6e5"
TEAL = "#127767"
ORANGE = "#ad5128"


def _evaluation_scores(path):
    scores = pd.read_csv(path, parse_dates=["forecast_start", "forecast_end"])
    models = [*ABLATION, FOUNDATION]
    selected = scores.loc[
        scores["role"].eq("evaluation") & scores["model"].isin(models)
    ]
    keys = ["forecast_start", "forecast_end", "fold"]
    rmsle = selected.pivot(index=keys, columns="model", values="rmsle").reindex(
        columns=models
    )
    counts = selected.pivot(index=keys, columns="model", values="n").reindex(
        columns=models
    )
    if (
        rmsle.empty
        or not np.isfinite(rmsle).to_numpy().all()
        or not counts.gt(0).to_numpy().all()
        or not counts.eq(counts[SCHEDULE], axis="index").to_numpy().all()
    ):
        raise ValueError(
            "Expected all four comparison models on the same evaluation rows"
        )
    pooled = ((rmsle.pow(2) * counts).sum() / counts.sum()).pow(0.5)
    return rmsle, pooled, int(counts[SCHEDULE].sum())


def _window_label(start, end):
    start_day = f"{start:%b} {start.day}"
    end_day = str(end.day) if start.month == end.month else f"{end:%b} {end.day}"
    return f"{start_day}–{end_day}"


def _panel_heading(ax, title, subtitle):
    ax.text(
        0, 1.22, title, transform=ax.transAxes, fontsize=14, weight="bold", color=INK
    )
    ax.text(0, 1.12, subtitle, transform=ax.transAxes, fontsize=10.5, color=MUTED)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    ax.tick_params(length=0, pad=7, labelcolor=MUTED, labelsize=9.5)
    ax.set_axisbelow(True)


def plot(results, output):
    """Plot matched scores by window and the fixed-settings promotion ablation."""
    scores, pooled, n = _evaluation_scores(results / "scores.csv")
    fig, (by_window, ablation) = plt.subplots(
        1, 2, figsize=(12.8, 4.8), facecolor="white"
    )
    fig.subplots_adjust(left=0.085, right=0.96, bottom=0.24, top=0.66, wspace=0.53)
    fig.text(
        0.085,
        0.93,
        "Forecast accuracy with scheduled promotions",
        fontsize=22,
        weight="bold",
        color=INK,
    )
    fig.text(
        0.085,
        0.865,
        "16-day store–family forecasts · RMSLE, lower is better",
        fontsize=12,
        color=MUTED,
    )

    _panel_heading(
        by_window,
        "Accuracy by evaluation window",
        "Both models receive the scheduled counts",
    )
    x = np.arange(len(scores))
    for model, label, color, marker, offset in (
        (SCHEDULE, "Pooled HGB", TEAL, "o", -16),
        (FOUNDATION, "t0-beta", ORANGE, "D", 9),
    ):
        series = scores[model]
        by_window.plot(
            x,
            series,
            label=label,
            color=color,
            marker=marker,
            linewidth=2.2,
            markersize=6,
        )
        for pos, value in zip(x, series, strict=True):
            by_window.annotate(
                f"{value:.5f}",
                (pos, value),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center",
                fontsize=9.5,
                weight="bold",
                color=color,
            )
    by_window.set_xticks(
        x, [_window_label(start, end) for start, end, _ in scores.index]
    )
    by_window.set_xlim(-0.23, len(scores) - 0.77)
    by_window.margins(y=0.25)
    by_window.grid(axis="y", color=RULE, linewidth=0.8)
    by_window.yaxis.set_major_locator(MaxNLocator(nbins=4))
    by_window.yaxis.set_major_formatter(StrMethodFormatter("{x:.3f}"))
    by_window.legend(
        loc="upper left",
        bbox_to_anchor=(0, -0.21),
        ncol=2,
        frameon=False,
        labelcolor=INK,
    )

    _panel_heading(
        ablation,
        "What promotion information adds",
        "Pooled HGB with the same settings in all three variants",
    )
    values = pooled[ABLATION]
    positions = np.arange(len(values))[::-1]
    spread = max(values.max() - values.min(), 0.006)
    ablation.set_xlim(values.min() - 0.41 * spread, values.max() + 0.53 * spread)
    ablation.set_ylim(-0.55, 2.55)
    ablation.set_yticks(
        positions, ["Sales + calendar", "+ Past promotions", "+ Scheduled counts"]
    )
    ablation.get_yticklabels()[-1].set_color(TEAL)
    ablation.get_yticklabels()[-1].set_weight("bold")
    ablation.grid(axis="both", color=RULE, linewidth=0.8)
    ablation.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ablation.xaxis.set_major_formatter(StrMethodFormatter("{x:.3f}"))
    ablation.set_xlabel("Pooled evaluation RMSLE", color=MUTED, fontsize=10)
    for value, pos, color in zip(values, positions, (MUTED, MUTED, TEAL), strict=True):
        ablation.scatter(value, pos, color=color, s=53, zorder=3)
        ablation.annotate(
            f"{value:.5f}",
            (value, pos),
            xytext=(9, 0),
            textcoords="offset points",
            va="center",
            fontsize=10,
            weight="bold",
            color=color,
        )

    years = sorted(set(scores.index.get_level_values("forecast_start").year))
    periods = ", ".join(map(str, years))
    fig.text(
        0.085,
        0.055,
        f"Historical evaluation in {periods} · {n:,} forecast rows per model across all windows",
        fontsize=9.5,
        color=MUTED,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("files/final-results"))
    parser.add_argument(
        "--output", type=Path, default=Path("files/forecast-evidence.png")
    )
    args = parser.parse_args()
    plt.switch_backend("Agg")
    plot(args.results, args.output)


if __name__ == "__main__":
    main()
