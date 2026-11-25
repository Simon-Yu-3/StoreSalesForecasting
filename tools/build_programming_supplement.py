"""Build a two-page PDF from the executable programming sample and its outputs."""

import argparse
import ast
import csv
import hashlib
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples/programming_sample.py"
WIDTH, HEIGHT = 612, 792
LEFT, RIGHT = 36, 576
INK = colors.HexColor("#152638")
MUTED = colors.HexColor("#4E6070")
ACCENT = colors.HexColor("#00746C")
CODE_BG = colors.HexColor("#F3F6F8")
REPOSITORY = "https://github.com/Simon-Yu-3/StoreSalesForecasting"


def paragraph(page, text, x, y, width, *, size=9.1, leading=12, color=INK):
    """Draw measured, wrapping text and return the next available baseline."""
    page.setFont("Helvetica", size)
    page.setFillColor(color)
    words, line = text.split(), ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if line and pdfmetrics.stringWidth(candidate, "Helvetica", size) > width:
            page.drawString(x, y, line)
            y -= leading
            line = word
        else:
            line = candidate
    if line:
        page.drawString(x, y, line)
        y -= leading
    return y


def source_rows():
    """Extract complete source functions rather than maintaining a second example."""
    content = SOURCE.read_text(encoding="utf-8")
    lines = content.splitlines()
    tree = ast.parse(content)
    definitions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }

    def function(name):
        node = definitions[name]
        return [(i + 1, lines[i]) for i in range(node.lineno - 1, node.end_lineno)]

    first = []
    for node in tree.body:
        if isinstance(node, ast.Import) or (
            isinstance(node, ast.ImportFrom) and node.module == "sklearn.ensemble"
        ):
            if isinstance(node, ast.Import) and node.names[0].name == "argparse":
                continue
            first.extend(
                (i + 1, lines[i]) for i in range(node.lineno - 1, node.end_lineno)
            )
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name)
            and target.id in {"KEYS", "HORIZON", "FEATURES"}
            for target in node.targets
        ):
            first.extend(
                (i + 1, lines[i]) for i in range(node.lineno - 1, node.end_lineno)
            )
    first += [
        (None, ""),
        *function("read_sales"),
        (None, ""),
        *function("build_features"),
    ]
    second = function("forecast")
    evaluation = definitions["evaluate"]
    plot_start = next(
        i
        for i in range(evaluation.lineno - 1, evaluation.end_lineno)
        if lines[i].lstrip().startswith("daily = predicted.groupby")
    )
    plot_end = next(
        i for i in range(plot_start, evaluation.end_lineno) if "plt.close" in lines[i]
    )
    call = evaluation.body[0]
    plotting = [(call.lineno, lines[call.lineno - 1][4:])]
    plotting.extend((i + 1, lines[i][4:]) for i in range(plot_start, plot_end + 1))
    return first, second, plotting


def code_panel(page, rows, top, *, size=9.0, leading=10.0):
    height = len(rows) * leading + 18
    page.setFillColor(CODE_BG)
    page.roundRect(LEFT, top - height, RIGHT - LEFT, height, 5, fill=1, stroke=0)
    baseline = top - 14
    for number, line in rows:
        if number is not None:
            page.setFillColor(MUTED)
            page.setFont("Courier", 7.0)
            page.drawRightString(LEFT + 22, baseline, str(number))
        page.setFont("Courier", size)
        page.setFillColor(MUTED if line.lstrip().startswith(("#", '"""')) else INK)
        if pdfmetrics.stringWidth(line, "Courier", size) > RIGHT - LEFT - 38:
            raise ValueError(f"Source line is too wide for the PDF: {number}")
        page.drawString(LEFT + 30, baseline, line)
        baseline -= leading
    return top - height


def header(page, number, heading):
    page.setFillColor(ACCENT)
    page.rect(LEFT, 754, RIGHT - LEFT, 2, fill=1, stroke=0)
    page.setFillColor(INK)
    page.setFont("Helvetica-Bold", 18)
    page.drawString(LEFT, 727, heading)
    page.setFont("Helvetica", 9)
    page.setFillColor(MUTED)
    page.drawString(LEFT, 709, "Junhui (Simon) Yu  |  Python programming supplement")
    page.drawRightString(RIGHT, 709, f"{number} / 2")


def footer(page, text):
    paragraph(page, text, LEFT, 63, RIGHT - LEFT, size=8.0, leading=10)
    page.setFillColor(ACCENT)
    page.setFont("Helvetica", 7.9)
    label = "Full executable code and experiment records: github.com/Simon-Yu-3/StoreSalesForecasting"
    page.drawString(LEFT, 29, label)
    page.linkURL(REPOSITORY, (LEFT, 25, RIGHT, 38), relative=0)


def build(output, demo_directory):
    first, second, plotting = source_rows()
    with (demo_directory / "metrics.csv").open(newline="", encoding="utf-8") as file:
        metrics = next(csv.DictReader(file))
    if metrics["dataset"] != "Synthetic demonstration":
        raise ValueError(
            "The illustration must be labeled synthetic demonstration data."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    page = canvas.Canvas(str(output), pagesize=(WIDTH, HEIGHT), pageCompression=1)
    page.setTitle("Programming Supplement - Store Sales Forecasting")
    page.setAuthor("Junhui (Simon) Yu")
    page.setSubject(
        "Executable Python excerpts with a chronological forecasting example"
    )
    source_digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    page.setKeywords(
        f"Python, forecasting, CSV, temporal validation, source SHA256 {source_digest}"
    )

    header(page, 1, "Store sales forecasting")
    y = paragraph(
        page,
        "The full project forecasts 16 days ahead using 3,000,888 historical sales "
        "observations across 54 stores and 33 product families. Its HGB/LightGBM/t0-beta "
        "ensemble reduced recorded Kaggle RMSLE from 0.47212 to 0.42172, a 10.68% "
        "reduction on the same 28,512 test rows.",
        LEFT,
        685,
        RIGHT - LEFT,
        size=9.3,
        leading=12,
    )
    y = paragraph(
        page,
        "This compact HGB example demonstrates typed CSV ingestion, schema checks, "
        "and calendar-aligned features. It runs on generated data by default "
        "and also accepts the official Kaggle training ZIP.",
        LEFT,
        y - 3,
        RIGHT - LEFT,
        size=9.0,
        leading=11.5,
    )
    page.setFillColor(ACCENT)
    page.setFont("Helvetica-Bold", 9.5)
    page.drawString(
        LEFT, y - 5, "CSV ingestion and calendar-aligned feature construction"
    )
    bottom = code_panel(page, first, y - 15, size=9.5, leading=10.8)
    if bottom < 91:
        raise ValueError("First-page code overlaps the attribution footer.")
    footer(
        page,
        "Attribution. This course project used AI-assisted coding, including Codex. "
        "The repository records that assistance and credits third-party estimators "
        "and the pretrained t0-beta checkpoint.",
    )
    page.showPage()

    header(page, 2, "Forecast from one fixed cutoff")
    y = paragraph(
        page,
        "A single cutoff reserves all 16 forecast dates. The 28-day lag is known "
        "throughout that horizon. Future promotion counts are assumed to be scheduled.",
        LEFT,
        685,
        RIGHT - LEFT,
        size=9.0,
        leading=11.5,
    )
    shown = (
        second
        + [(None, "# Forecast and visualization excerpt from evaluate()")]
        + plotting[:3]
    )
    bottom = code_panel(page, shown, y - 4, size=9.3, leading=9.5)
    plot_top = bottom - 14
    page.setFont("Helvetica-Bold", 8.6)
    page.setFillColor(INK)
    page.drawString(
        LEFT,
        plot_top,
        "Synthetic run: 6 series, 96 holdout rows"
        f"  |  RMSLE {float(metrics['rmsle']):.5f}"
        f"  |  MAE {float(metrics['mae']):.2f} sales units",
    )
    page.drawImage(
        ImageReader(str(demo_directory / "forecast.png")),
        LEFT,
        plot_top - 209.5,
        width=540,
        height=202.5,
        mask="auto",
    )
    paragraph(
        page,
        "Tests verify calendar-gap alignment and unchanged forecasts after replacing all held-out sales.",
        LEFT,
        79,
        RIGHT - LEFT,
        size=8.3,
        leading=10,
    )
    if plot_top - 209.5 < 88:
        raise ValueError("Second-page illustration overlaps the scope footer.")
    footer(
        page,
        "Scope. The plotted run uses synthetic data. The full project's public "
        "Kaggle score was observed during iterative development; later historical "
        "checks reuse previously reviewed dates.",
    )
    page.save()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "output/pdf/StoreSalesForecasting-Programming-Supplement.pdf",
    )
    parser.add_argument(
        "--demo-dir",
        type=Path,
        default=ROOT / "results/programming-sample",
    )
    args = parser.parse_args()
    print(build(args.output, args.demo_dir))


if __name__ == "__main__":
    main()
