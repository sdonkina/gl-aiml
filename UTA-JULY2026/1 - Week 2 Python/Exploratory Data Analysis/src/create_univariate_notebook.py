from __future__ import annotations

import base64
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import median


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "customer_level_analysis_dataset.csv"
NOTEBOOK_PATH = BASE_DIR / "notebooks" / "01_univariate_analysis.ipynb"


def read_rows() -> list[dict[str, str]]:
    with DATA_PATH.open(newline="", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def svg_data_uri(svg: str) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")


def wrap_svg(content: str, width: int = 920, height: int = 460) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="#f8f6f2"/>'
        f"{content}</svg>"
    )


def escape_xml(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def bar_chart_svg(title: str, labels: list[str], values: list[float], color: str) -> str:
    width = 920
    height = 460
    left = 80
    right = 30
    top = 70
    bottom = 120
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_val = max(values) if values else 1
    max_val = max_val or 1
    bar_gap = 18
    bar_w = max((plot_w - bar_gap * (len(values) - 1)) / max(len(values), 1), 20)

    parts = [
        f'<text x="{left}" y="38" font-size="24" font-family="Helvetica, Arial, sans-serif" '
        f'font-weight="700" fill="#1f2937">{escape_xml(title)}</text>'
    ]

    for step in range(5):
        value = max_val * step / 4
        y = top + plot_h - (plot_h * step / 4)
        parts.append(f'<line x1="{left}" y1="{y}" x2="{left + plot_w}" y2="{y}" stroke="#ddd6c8" stroke-width="1"/>')
        parts.append(
            f'<text x="{left - 12}" y="{y + 5}" text-anchor="end" font-size="12" '
            f'font-family="Helvetica, Arial, sans-serif" fill="#6b7280">{int(round(value))}</text>'
        )

    for idx, (label, value) in enumerate(zip(labels, values)):
        x = left + idx * (bar_w + bar_gap)
        bar_h = 0 if max_val == 0 else (value / max_val) * plot_h
        y = top + plot_h - bar_h
        parts.append(
            f'<rect x="{x}" y="{y}" width="{bar_w}" height="{bar_h}" rx="6" fill="{color}" opacity="0.92"/>'
        )
        parts.append(
            f'<text x="{x + bar_w / 2}" y="{y - 8}" text-anchor="middle" font-size="12" '
            f'font-family="Helvetica, Arial, sans-serif" fill="#374151">{value}</text>'
        )
        parts.append(
            f'<text x="{x + bar_w / 2}" y="{top + plot_h + 20}" text-anchor="middle" font-size="12" '
            f'font-family="Helvetica, Arial, sans-serif" fill="#374151">{escape_xml(label)}</text>'
        )

    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#6b7280"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#6b7280"/>')
    return wrap_svg("".join(parts), width, height)


def histogram_svg(title: str, values: list[float], bins: int, color: str) -> str:
    width = 920
    height = 460
    left = 80
    right = 30
    top = 70
    bottom = 90
    plot_w = width - left - right
    plot_h = height - top - bottom

    min_val = min(values)
    max_val = max(values)
    if math.isclose(min_val, max_val):
        edges = [min_val, max_val + 1]
    else:
        bin_width = (max_val - min_val) / bins
        edges = [min_val + i * bin_width for i in range(bins)] + [max_val]

    counts = [0 for _ in range(bins)]
    if math.isclose(min_val, max_val):
        counts[0] = len(values)
    else:
        width_val = (max_val - min_val) / bins
        for value in values:
            idx = min(int((value - min_val) / width_val), bins - 1)
            counts[idx] += 1

    max_count = max(counts) if counts else 1
    bar_gap = 4
    bar_w = (plot_w - bar_gap * (bins - 1)) / bins

    parts = [
        f'<text x="{left}" y="38" font-size="24" font-family="Helvetica, Arial, sans-serif" '
        f'font-weight="700" fill="#1f2937">{escape_xml(title)}</text>'
    ]

    for step in range(5):
        value = max_count * step / 4
        y = top + plot_h - (plot_h * step / 4)
        parts.append(f'<line x1="{left}" y1="{y}" x2="{left + plot_w}" y2="{y}" stroke="#ddd6c8" stroke-width="1"/>')
        parts.append(
            f'<text x="{left - 12}" y="{y + 5}" text-anchor="end" font-size="12" '
            f'font-family="Helvetica, Arial, sans-serif" fill="#6b7280">{int(round(value))}</text>'
        )

    for idx, count in enumerate(counts):
        x = left + idx * (bar_w + bar_gap)
        bar_h = 0 if max_count == 0 else (count / max_count) * plot_h
        y = top + plot_h - bar_h
        parts.append(
            f'<rect x="{x}" y="{y}" width="{bar_w}" height="{bar_h}" fill="{color}" opacity="0.88"/>'
        )

    x_labels = [min_val, (min_val + max_val) / 2, max_val]
    x_positions = [left, left + plot_w / 2, left + plot_w]
    for pos, label in zip(x_positions, x_labels):
        parts.append(
            f'<text x="{pos}" y="{top + plot_h + 24}" text-anchor="middle" font-size="12" '
            f'font-family="Helvetica, Arial, sans-serif" fill="#374151">{round(label, 1)}</text>'
        )

    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#6b7280"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#6b7280"/>')
    return wrap_svg("".join(parts), width, height)


def summary_markdown(rows: list[dict[str, str]]) -> str:
    total_customers = len(rows)
    total_orders = sum(int(row["total_orders"]) for row in rows)
    avg_orders = round(total_orders / total_customers, 2)
    median_orders = median(int(row["total_orders"]) for row in rows)
    repeat_rate = round(
        sum(int(row["repeat_customer_flag"]) for row in rows) / total_customers * 100,
        1,
    )
    avg_rating = round(
        sum(float(row["avg_customer_rating"]) for row in rows) / total_customers,
        2,
    )
    return "\n".join(
        [
            "## Dataset Snapshot",
            f"- Customers: `{total_customers}`",
            f"- Total orders represented: `{total_orders}`",
            f"- Average orders per customer: `{avg_orders}`",
            f"- Median orders per customer: `{median_orders}`",
            f"- Repeat-customer rate: `{repeat_rate}%`",
            f"- Average customer rating: `{avg_rating}`",
        ]
    )


def cohort_retention_markdown(rows: list[dict[str, str]], cohort_field: str, label: str) -> str:
    cohort_stats: dict[str, dict[str, int]] = {}
    for row in rows:
        cohort = row[cohort_field][:7]
        cohort_stats.setdefault(
            cohort,
            {
                "customers": 0,
                "r30_num": 0,
                "r30_den": 0,
                "r60_num": 0,
                "r60_den": 0,
                "r90_num": 0,
                "r90_den": 0,
            },
        )
        cohort_stats[cohort]["customers"] += 1
        for window in ["30", "60", "90"]:
            value = row[f"returned_after_first_order_{window}d"]
            if value != "":
                cohort_stats[cohort][f"r{window}_den"] += 1
                cohort_stats[cohort][f"r{window}_num"] += int(value)

    lines = [
        f"### {label} Cohort Retention Table",
        "",
        "| Cohort Month | Customers | 30d Observed | 30d Retention % | 60d Observed | 60d Retention % | 90d Observed | 90d Retention % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cohort in sorted(cohort_stats):
        stats = cohort_stats[cohort]
        r30 = round(stats["r30_num"] / stats["r30_den"] * 100, 1) if stats["r30_den"] else ""
        r60 = round(stats["r60_num"] / stats["r60_den"] * 100, 1) if stats["r60_den"] else ""
        r90 = round(stats["r90_num"] / stats["r90_den"] * 100, 1) if stats["r90_den"] else ""
        lines.append(
            f"| {cohort} | {stats['customers']} | {stats['r30_den']} | {r30} | "
            f"{stats['r60_den']} | {r60} | {stats['r90_den']} | {r90} |"
        )
    return "\n".join(lines)


def frequency_table_markdown(title: str, counts: list[tuple[str, int]], total: int) -> str:
    lines = [
        f"### {title}",
        "",
        "| Category | Count | Percentage |",
        "|---|---:|---:|",
    ]
    for label, count in counts:
        pct = round(count / total * 100, 1) if total else 0.0
        lines.append(f"| {label} | {count} | {pct}% |")
    return "\n".join(lines)


def line_chart_svg(
    title: str,
    labels: list[str],
    series: list[tuple[str, list[float], str]],
    y_min: float = 0.0,
    y_max: float = 100.0,
) -> str:
    width = 960
    height = 480
    left = 80
    right = 30
    top = 70
    bottom = 110
    plot_w = width - left - right
    plot_h = height - top - bottom

    def x_pos(idx: int) -> float:
        if len(labels) == 1:
            return left + plot_w / 2
        return left + idx * (plot_w / (len(labels) - 1))

    def y_pos(value: float) -> float:
        return top + plot_h - ((value - y_min) / (y_max - y_min) * plot_h)

    parts = [
        f'<text x="{left}" y="38" font-size="24" font-family="Helvetica, Arial, sans-serif" '
        f'font-weight="700" fill="#1f2937">{escape_xml(title)}</text>'
    ]

    for tick in range(6):
        value = y_min + (y_max - y_min) * tick / 5
        y = y_pos(value)
        parts.append(f'<line x1="{left}" y1="{y}" x2="{left + plot_w}" y2="{y}" stroke="#ddd6c8" stroke-width="1"/>')
        parts.append(
            f'<text x="{left - 12}" y="{y + 5}" text-anchor="end" font-size="12" '
            f'font-family="Helvetica, Arial, sans-serif" fill="#6b7280">{round(value, 0)}%</text>'
        )

    for idx, label in enumerate(labels):
        x = x_pos(idx)
        parts.append(
            f'<text x="{x}" y="{top + plot_h + 24}" text-anchor="middle" font-size="11" '
            f'font-family="Helvetica, Arial, sans-serif" fill="#374151">{escape_xml(label)}</text>'
        )

    legend_x = left
    legend_y = height - 28
    for name, values, color in series:
        path = []
        for idx, value in enumerate(values):
            command = "M" if idx == 0 else "L"
            path.append(f"{command} {x_pos(idx):.2f} {y_pos(value):.2f}")
        parts.append(f'<path d="{" ".join(path)}" fill="none" stroke="{color}" stroke-width="3"/>')
        for idx, value in enumerate(values):
            cx = x_pos(idx)
            cy = y_pos(value)
            parts.append(f'<circle cx="{cx}" cy="{cy}" r="4.5" fill="{color}"/>')
        parts.append(f'<rect x="{legend_x}" y="{legend_y - 10}" width="18" height="18" fill="{color}"/>')
        parts.append(
            f'<text x="{legend_x + 26}" y="{legend_y + 4}" font-size="12" '
            f'font-family="Helvetica, Arial, sans-serif" fill="#374151">{escape_xml(name)}</text>'
        )
        legend_x += 150

    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#6b7280"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#6b7280"/>')
    return wrap_svg("".join(parts), width, height)


def make_markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def make_code_cell(source: str, output_text: str | None = None, svg: str | None = None) -> dict:
    outputs = []
    if output_text is not None:
        outputs.append({"name": "stdout", "output_type": "stream", "text": output_text})
    if svg is not None:
        outputs.append(
            {
                "data": {"image/svg+xml": svg, "text/plain": ["<univariate plot>"]},
                "metadata": {},
                "output_type": "display_data",
            }
        )
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": outputs,
        "source": source,
    }


def build_notebook(rows: list[dict[str, str]]) -> dict:
    city_counts = Counter(row["city"] for row in rows).most_common()
    age_counts = Counter(row["age_group"] for row in rows)
    gender_counts = Counter(row["gender"] for row in rows)
    segment_counts = Counter(row["customer_segment"] for row in rows)
    device_counts = Counter(row["preferred_device"] for row in rows)
    total_customers = len(rows)

    total_orders = [int(row["total_orders"]) for row in rows]
    total_order_value = [float(row["total_order_value_usd"]) for row in rows]
    on_time_rates = [float(row["on_time_delivery_rate"]) for row in rows]
    avg_delays = [float(row["avg_delay_minutes"]) for row in rows]
    ratings = [float(row["avg_customer_rating"]) for row in rows]
    promo_rates = [float(row["promo_order_rate"]) for row in rows]
    tenure_days = [float(row["tenure_days_at_dataset_end"]) for row in rows]

    first_order_cohorts = sorted({row["first_order_date"][:7] for row in rows})
    cohort_30 = []
    cohort_60 = []
    cohort_90 = []
    for cohort in first_order_cohorts:
        cohort_rows = [row for row in rows if row["first_order_date"][:7] == cohort]
        for window, bucket in [("30", cohort_30), ("60", cohort_60), ("90", cohort_90)]:
            observed = [
                row[f"returned_after_first_order_{window}d"]
                for row in cohort_rows
                if row[f"returned_after_first_order_{window}d"] != ""
            ]
            rate = round(sum(int(value) for value in observed) / len(observed) * 100, 1) if observed else 0.0
            if window == "30":
                cohort_30.append(rate)
            elif window == "60":
                cohort_60.append(rate)
            else:
                cohort_90.append(rate)

    cells = [
        make_markdown_cell(
            "# 01 Univariate Analysis\n"
            "This notebook reviews the individual distribution of the most relevant variables in the processed customer-level dataset."
        ),
        make_markdown_cell(summary_markdown(rows)),
        make_code_cell(
            "from pathlib import Path\n"
            "import csv\n\n"
            "data_path = Path('../data/processed/customer_level_analysis_dataset.csv')\n"
            "with data_path.open(newline='', encoding='utf-8') as f:\n"
            "    rows = list(csv.DictReader(f))\n"
            "print(f'Rows: {len(rows)} | Columns: {len(rows[0])}')\n",
            output_text=f"Rows: {len(rows)} | Columns: {len(rows[0])}\n",
        ),
        make_markdown_cell("## Customer Profile Variables"),
        make_markdown_cell(frequency_table_markdown("City Frequency Distribution", city_counts, total_customers)),
        make_code_cell(
            "# Distribution of customers by city\n"
            "# Pre-rendered in this notebook for quick review.\n",
            svg=bar_chart_svg(
                "Customer Count by City",
                [label for label, _ in city_counts],
                [value for _, value in city_counts],
                "#b45309",
            ),
        ),
        make_code_cell(
            "# Distribution of customer age groups\n",
            svg=bar_chart_svg(
                "Customer Count by Age Group",
                ["18-25", "26-35", "36-45", "46-55", "56+"],
                [age_counts[label] for label in ["18-25", "26-35", "36-45", "46-55", "56+"]],
                "#047857",
            ),
        ),
        make_markdown_cell(
            frequency_table_markdown(
                "Age Group Frequency Distribution",
                [(label, age_counts[label]) for label in ["18-25", "26-35", "36-45", "46-55", "56+"]],
                total_customers,
            )
        ),
        make_code_cell(
            "# Distribution of gender\n",
            svg=bar_chart_svg(
                "Customer Count by Gender",
                ["Female", "Male", "Other"],
                [gender_counts[label] for label in ["Female", "Male", "Other"]],
                "#be185d",
            ),
        ),
        make_markdown_cell(
            frequency_table_markdown(
                "Gender Frequency Distribution",
                [(label, gender_counts[label]) for label in ["Female", "Male", "Other"]],
                total_customers,
            )
        ),
        make_code_cell(
            "# Distribution of customer segments\n",
            svg=bar_chart_svg(
                "Customer Count by Segment",
                ["New", "Regular", "Premium"],
                [segment_counts[label] for label in ["New", "Regular", "Premium"]],
                "#7c3aed",
            ),
        ),
        make_code_cell(
            "# Preferred device split\n",
            svg=bar_chart_svg(
                "Preferred Device",
                ["Mobile App", "Web"],
                [device_counts[label] for label in ["Mobile App", "Web"]],
                "#2563eb",
            ),
        ),
        make_markdown_cell(
            frequency_table_markdown(
                "Preferred Device Frequency Distribution",
                [(label, device_counts[label]) for label in ["Mobile App", "Web"]],
                total_customers,
            )
        ),
        make_markdown_cell("## Customer Behavior Variables"),
        make_code_cell(
            "# Distribution of total orders per customer\n",
            svg=histogram_svg("Total Orders per Customer", total_orders, bins=20, color="#0f766e"),
        ),
        make_code_cell(
            "# Distribution of total spend per customer\n",
            svg=histogram_svg("Total Order Value per Customer (USD)", total_order_value, bins=20, color="#dc2626"),
        ),
        make_code_cell(
            "# Distribution of customer tenure in days\n",
            svg=histogram_svg("Customer Tenure at Dataset End (Days)", tenure_days, bins=20, color="#1d4ed8"),
        ),
        make_code_cell(
            "# Distribution of promotion usage rate\n",
            svg=histogram_svg("Promotion Order Rate", promo_rates, bins=20, color="#9333ea"),
        ),
        make_markdown_cell("## Delivery Experience Variables"),
        make_code_cell(
            "# Distribution of on-time delivery rate\n",
            svg=histogram_svg("On-Time Delivery Rate", on_time_rates, bins=20, color="#059669"),
        ),
        make_code_cell(
            "# Distribution of average delivery delay\n",
            svg=histogram_svg("Average Delay Minutes", avg_delays, bins=20, color="#ea580c"),
        ),
        make_code_cell(
            "# Distribution of average customer rating\n",
            svg=histogram_svg("Average Customer Rating", ratings, bins=20, color="#ca8a04"),
        ),
        make_markdown_cell("## Retention by Cohort"),
        make_markdown_cell(
            "The tables and plot below show how first-order retention varies by monthly first-order cohort. "
            "Only observed customers are included in each retention window."
        ),
        make_markdown_cell(cohort_retention_markdown(rows, "first_order_date", "First-Order")),
        make_code_cell(
            "# Monthly first-order cohort retention trends\n",
            svg=line_chart_svg(
                "Retention by First-Order Cohort",
                first_order_cohorts,
                [
                    ("30-day retention", cohort_30, "#b45309"),
                    ("60-day retention", cohort_60, "#2563eb"),
                    ("90-day retention", cohort_90, "#059669"),
                ],
                y_min=70.0,
                y_max=100.0,
            ),
        ),
        make_markdown_cell(
            "## Notes\n"
            "- The plots are embedded directly in the notebook so you can review them immediately.\n"
            "- If you later install `pandas` and `matplotlib`, we can swap these lightweight visuals for fully programmable plotting cells."
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.9",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    rows = read_rows()
    notebook = build_notebook(rows)
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    print(f"Created notebook at {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
