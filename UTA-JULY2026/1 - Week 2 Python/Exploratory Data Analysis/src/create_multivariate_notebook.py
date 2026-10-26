from __future__ import annotations

import csv
import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "customer_level_analysis_dataset.csv"
NOTEBOOK_PATH = BASE_DIR / "notebooks" / "03_multivariate_analysis.ipynb"


def read_rows() -> list[dict[str, str]]:
    with DATA_PATH.open(newline="", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def escape_xml(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def wrap_svg(content: str, width: int = 980, height: int = 540) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="#f8f6f2"/>'
        f"{content}</svg>"
    )


def retention_rate(rows: list[dict[str, str]], window: str = "60") -> float:
    values = [row[f"returned_after_first_order_{window}d"] for row in rows if row[f"returned_after_first_order_{window}d"] != ""]
    return round(sum(int(value) for value in values) / len(values) * 100, 1) if values else 0.0


def sample_size(rows: list[dict[str, str]], window: str = "60") -> int:
    return sum(1 for row in rows if row[f"returned_after_first_order_{window}d"] != "")


def bin_on_time_rate(value: float) -> str:
    if value < 0.50:
        return "< 50%"
    if value < 0.70:
        return "50%-69%"
    if value < 0.85:
        return "70%-84%"
    if value < 0.95:
        return "85%-94%"
    return "95%-100%"


def bin_avg_delay(value: float) -> str:
    if value <= 2:
        return "0-2 min"
    if value <= 5:
        return "2-5 min"
    if value <= 10:
        return "5-10 min"
    if value <= 15:
        return "10-15 min"
    return "15+ min"


def bin_rating(value: float) -> str:
    if value < 3.0:
        return "1.0-2.9"
    if value < 3.5:
        return "3.0-3.4"
    if value < 4.0:
        return "3.5-3.9"
    if value < 4.5:
        return "4.0-4.4"
    return "4.5-5.0"


def bin_promo_rate(value: float) -> str:
    if value == 0:
        return "0%"
    if value <= 0.25:
        return "1%-25%"
    if value <= 0.50:
        return "26%-50%"
    if value <= 0.75:
        return "51%-75%"
    return "76%-100%"


def category_breadth_label(value: str) -> str:
    return f"{value} category" if value == "1" else f"{value} categories"


def service_breadth_label(value: str) -> str:
    return f"{value} type" if value == "1" else f"{value} types"


def heatmap_svg(
    title: str,
    row_labels: list[str],
    col_labels: list[str],
    matrix: list[list[float]],
    count_matrix: list[list[int]] | None = None,
) -> str:
    width = 980
    height = 540
    left = 220
    right = 50
    top = 100
    bottom = 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    cell_w = plot_w / max(len(col_labels), 1)
    cell_h = plot_h / max(len(row_labels), 1)

    flat = [value for row in matrix for value in row]
    min_v = min(flat) if flat else 0
    max_v = max(flat) if flat else 100

    def color(value: float) -> str:
        ratio = 0.5 if max_v == min_v else (value - min_v) / (max_v - min_v)
        r = int(243 - 120 * ratio)
        g = int(229 - 130 * ratio)
        b = int(211 - 160 * ratio)
        return f"rgb({r},{g},{b})"

    parts = [
        f'<text x="{left}" y="42" font-size="24" font-family="Helvetica, Arial, sans-serif" font-weight="700" fill="#1f2937">{escape_xml(title)}</text>',
        f'<text x="{left}" y="70" font-size="12" font-family="Helvetica, Arial, sans-serif" fill="#6b7280">Cell values show 60-day retention %. Sample size shown below each cell value.</text>',
    ]

    for c_idx, label in enumerate(col_labels):
        x = left + c_idx * cell_w + cell_w / 2
        parts.append(
            f'<text x="{x}" y="{top - 22}" text-anchor="middle" font-size="12" font-family="Helvetica, Arial, sans-serif" fill="#374151">{escape_xml(label)}</text>'
        )

    for r_idx, label in enumerate(row_labels):
        y = top + r_idx * cell_h + cell_h / 2 + 5
        parts.append(
            f'<text x="{left - 14}" y="{y}" text-anchor="end" font-size="12" font-family="Helvetica, Arial, sans-serif" fill="#374151">{escape_xml(label)}</text>'
        )
        for c_idx, value in enumerate(matrix[r_idx]):
            x = left + c_idx * cell_w
            y0 = top + r_idx * cell_h
            parts.append(f'<rect x="{x}" y="{y0}" width="{cell_w}" height="{cell_h}" fill="{color(value)}" stroke="#fff" stroke-width="2"/>')
            parts.append(
                f'<text x="{x + cell_w / 2}" y="{y0 + cell_h / 2 - 2}" text-anchor="middle" font-size="14" font-family="Helvetica, Arial, sans-serif" font-weight="700" fill="#111827">{value:.1f}%</text>'
            )
            if count_matrix is not None:
                parts.append(
                    f'<text x="{x + cell_w / 2}" y="{y0 + cell_h / 2 + 18}" text-anchor="middle" font-size="11" font-family="Helvetica, Arial, sans-serif" fill="#374151">n={count_matrix[r_idx][c_idx]}</text>'
                )
    return wrap_svg("".join(parts), width, height)


def grouped_bar_chart_svg(
    title: str,
    labels: list[str],
    series: list[tuple[str, list[float], str]],
    y_min: float = 60.0,
    y_max: float = 100.0,
) -> str:
    width = 980
    height = 520
    left = 90
    right = 30
    top = 80
    bottom = 130
    plot_w = width - left - right
    plot_h = height - top - bottom
    group_gap = 24
    inner_gap = 6
    group_w = (plot_w - group_gap * (len(labels) - 1)) / max(len(labels), 1)
    bar_w = (group_w - inner_gap * (len(series) - 1)) / max(len(series), 1)

    parts = [
        f'<text x="{left}" y="40" font-size="24" font-family="Helvetica, Arial, sans-serif" font-weight="700" fill="#1f2937">{escape_xml(title)}</text>'
    ]

    for tick in range(6):
        value = y_min + (y_max - y_min) * tick / 5
        y = top + plot_h - ((value - y_min) / (y_max - y_min) * plot_h)
        parts.append(f'<line x1="{left}" y1="{y}" x2="{left + plot_w}" y2="{y}" stroke="#ddd6c8" stroke-width="1"/>')
        parts.append(
            f'<text x="{left - 12}" y="{y + 5}" text-anchor="end" font-size="12" font-family="Helvetica, Arial, sans-serif" fill="#6b7280">{round(value, 0)}%</text>'
        )

    for idx, label in enumerate(labels):
        group_x = left + idx * (group_w + group_gap)
        for s_idx, (_, values, color) in enumerate(series):
            value = values[idx]
            x = group_x + s_idx * (bar_w + inner_gap)
            bar_h = ((value - y_min) / (y_max - y_min) * plot_h) if y_max > y_min else 0
            y = top + plot_h - bar_h
            parts.append(f'<rect x="{x}" y="{y}" width="{bar_w}" height="{bar_h}" rx="4" fill="{color}" opacity="0.92"/>')
            parts.append(
                f'<text x="{x + bar_w / 2}" y="{y - 8}" text-anchor="middle" font-size="11" font-family="Helvetica, Arial, sans-serif" fill="#374151">{value:.1f}</text>'
            )
        parts.append(
            f'<text x="{group_x + group_w / 2}" y="{top + plot_h + 22}" text-anchor="middle" font-size="11" font-family="Helvetica, Arial, sans-serif" fill="#374151">{escape_xml(label)}</text>'
        )

    legend_x = left
    legend_y = height - 30
    for name, _, color in series:
        parts.append(f'<rect x="{legend_x}" y="{legend_y - 10}" width="18" height="18" fill="{color}"/>')
        parts.append(
            f'<text x="{legend_x + 26}" y="{legend_y + 4}" font-size="12" font-family="Helvetica, Arial, sans-serif" fill="#374151">{escape_xml(name)}</text>'
        )
        legend_x += 170

    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#6b7280"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#6b7280"/>')
    return wrap_svg("".join(parts), width, height)


def interaction_table_markdown(
    title: str,
    row_labels: list[str],
    col_labels: list[str],
    matrix: list[list[float]],
    count_matrix: list[list[int]],
) -> str:
    lines = [f"### {title}", "", "| Group | " + " | ".join(col_labels) + " |", "|" + "---|" * (len(col_labels) + 1)]
    for label, row_values, row_counts in zip(row_labels, matrix, count_matrix):
        values = [f"{value:.1f}% (n={count})" for value, count in zip(row_values, row_counts)]
        lines.append(f"| {label} | " + " | ".join(values) + " |")
    return "\n".join(lines)


def make_markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def make_code_cell(source: str, output_text: str | None = None, svg: str | None = None) -> dict:
    outputs = []
    if output_text is not None:
        outputs.append({"name": "stdout", "output_type": "stream", "text": output_text})
    if svg is not None:
        outputs.append(
            {
                "data": {"image/svg+xml": svg, "text/plain": ["<multivariate plot>"]},
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


def build_matrix(
    rows: list[dict[str, str]],
    row_labels: list[str],
    col_labels: list[str],
    row_fn,
    col_fn,
    window: str = "60",
) -> tuple[list[list[float]], list[list[int]]]:
    matrix = []
    count_matrix = []
    for r_label in row_labels:
        matrix_row = []
        count_row = []
        for c_label in col_labels:
            subset = [row for row in rows if row_fn(row) == r_label and col_fn(row) == c_label]
            matrix_row.append(retention_rate(subset, window))
            count_row.append(sample_size(subset, window))
        matrix.append(matrix_row)
        count_matrix.append(count_row)
    return matrix, count_matrix


def build_notebook(rows: list[dict[str, str]]) -> dict:
    on_time_labels = ["< 50%", "50%-69%", "70%-84%", "85%-94%", "95%-100%"]
    rating_labels = ["1.0-2.9", "3.0-3.4", "3.5-3.9", "4.0-4.4", "4.5-5.0"]
    delay_labels = ["0-2 min", "2-5 min", "5-10 min", "10-15 min", "15+ min"]
    promo_labels = ["0%", "1%-25%", "26%-50%", "51%-75%", "76%-100%"]
    segment_labels = ["New", "Regular", "Premium"]
    device_labels = ["Mobile App", "Web"]
    category_labels = ["1 category", "2 categories", "3 categories", "4 categories"]
    service_labels = ["1 type", "2 types", "3 types"]

    reliability_matrix, reliability_counts = build_matrix(
        rows,
        rating_labels,
        on_time_labels,
        lambda row: bin_rating(float(row["avg_customer_rating"])),
        lambda row: bin_on_time_rate(float(row["on_time_delivery_rate"])),
    )
    delay_segment_matrix, delay_segment_counts = build_matrix(
        rows,
        delay_labels,
        segment_labels,
        lambda row: bin_avg_delay(float(row["avg_delay_minutes"])),
        lambda row: row["customer_segment"],
    )
    promo_segment_matrix, promo_segment_counts = build_matrix(
        rows,
        promo_labels,
        segment_labels,
        lambda row: bin_promo_rate(float(row["promo_order_rate"])),
        lambda row: row["customer_segment"],
    )
    promo_device_matrix, promo_device_counts = build_matrix(
        rows,
        promo_labels,
        device_labels,
        lambda row: bin_promo_rate(float(row["promo_order_rate"])),
        lambda row: row["preferred_device"],
    )
    breadth_matrix, breadth_counts = build_matrix(
        rows,
        category_labels,
        service_labels,
        lambda row: category_breadth_label(row["unique_vendor_categories"]),
        lambda row: service_breadth_label(row["unique_delivery_types"]),
    )
    breadth_segment_matrix, breadth_segment_counts = build_matrix(
        rows,
        category_labels,
        segment_labels,
        lambda row: category_breadth_label(row["unique_vendor_categories"]),
        lambda row: row["customer_segment"],
    )

    segment_series_labels = ["New", "Regular", "Premium"]
    category_segment_rates = []
    for segment in segment_series_labels:
        category_segment_rates.append(
            [
                retention_rate(
                    [
                        row for row in rows
                        if row["customer_segment"] == segment
                        and category_breadth_label(row["unique_vendor_categories"]) == category_label
                    ],
                    "60",
                )
                for category_label in category_labels
            ]
        )

    cells = [
        make_markdown_cell(
            "# 03 Multivariate Analysis\n"
            "This notebook adds multivariate, interaction-focused visuals for the project hypotheses. "
            "It stays strictly descriptive and does not run any statistical hypothesis tests."
        ),
        make_markdown_cell(
            "## Scope\n"
            "- Hypothesis 1: delivery reliability and retention, viewed jointly across multiple experience dimensions\n"
            "- Hypothesis 2: promotion behavior and retention, stratified by customer type and device\n"
            "- Hypothesis 3: breadth of platform usage and retention, viewed jointly with service breadth and segment\n"
            "- All visuals show observed `60-day` retention unless otherwise stated"
        ),
        make_code_cell(
            "from pathlib import Path\n"
            "import csv\n\n"
            "data_path = Path('../data/processed/customer_level_analysis_dataset.csv')\n"
            "with data_path.open(newline='', encoding='utf-8') as f:\n"
            "    rows = list(csv.DictReader(f))\n"
            "print(f'Rows: {len(rows)} | Columns: {len(rows[0])}')\n",
            output_text=f"Rows: {len(rows)} | Columns: {len(rows[0])}\n",
        ),
        make_markdown_cell("## Hypothesis 1: Delivery Reliability vs Retention"),
        make_markdown_cell(
            "This section looks at retention after jointly considering multiple delivery-experience signals rather than one variable at a time."
        ),
        make_markdown_cell(
            interaction_table_markdown(
                "60-Day Retention by Customer Rating and On-Time Delivery Rate",
                rating_labels,
                on_time_labels,
                reliability_matrix,
                reliability_counts,
            )
        ),
        make_code_cell(
            "# Joint view of customer rating and on-time delivery rate\n",
            svg=heatmap_svg(
                "60-Day Retention by Customer Rating and On-Time Delivery Rate",
                rating_labels,
                on_time_labels,
                reliability_matrix,
                reliability_counts,
            ),
        ),
        make_markdown_cell(
            interaction_table_markdown(
                "60-Day Retention by Average Delay and Customer Segment",
                delay_labels,
                segment_labels,
                delay_segment_matrix,
                delay_segment_counts,
            )
        ),
        make_code_cell(
            "# Average delay stratified by customer segment\n",
            svg=heatmap_svg(
                "60-Day Retention by Average Delay and Customer Segment",
                delay_labels,
                segment_labels,
                delay_segment_matrix,
                delay_segment_counts,
            ),
        ),
        make_markdown_cell("## Hypothesis 2: Promotion Behavior vs Retention"),
        make_markdown_cell(
            interaction_table_markdown(
                "60-Day Retention by Promotion Usage Rate and Customer Segment",
                promo_labels,
                segment_labels,
                promo_segment_matrix,
                promo_segment_counts,
            )
        ),
        make_code_cell(
            "# Promotion intensity stratified by customer segment\n",
            svg=heatmap_svg(
                "60-Day Retention by Promotion Usage Rate and Customer Segment",
                promo_labels,
                segment_labels,
                promo_segment_matrix,
                promo_segment_counts,
            ),
        ),
        make_markdown_cell(
            interaction_table_markdown(
                "60-Day Retention by Promotion Usage Rate and Preferred Device",
                promo_labels,
                device_labels,
                promo_device_matrix,
                promo_device_counts,
            )
        ),
        make_code_cell(
            "# Promotion intensity stratified by preferred device\n",
            svg=heatmap_svg(
                "60-Day Retention by Promotion Usage Rate and Preferred Device",
                promo_labels,
                device_labels,
                promo_device_matrix,
                promo_device_counts,
            ),
        ),
        make_markdown_cell("## Hypothesis 3: Breadth of Platform Usage vs Retention"),
        make_markdown_cell(
            interaction_table_markdown(
                "60-Day Retention by Vendor-Category Breadth and Delivery-Type Breadth",
                category_labels,
                service_labels,
                breadth_matrix,
                breadth_counts,
            )
        ),
        make_code_cell(
            "# Joint view of category breadth and service breadth\n",
            svg=heatmap_svg(
                "60-Day Retention by Vendor-Category Breadth and Delivery-Type Breadth",
                category_labels,
                service_labels,
                breadth_matrix,
                breadth_counts,
            ),
        ),
        make_markdown_cell(
            interaction_table_markdown(
                "60-Day Retention by Vendor-Category Breadth and Customer Segment",
                category_labels,
                segment_labels,
                breadth_segment_matrix,
                breadth_segment_counts,
            )
        ),
        make_code_cell(
            "# Category breadth stratified by customer segment\n",
            svg=heatmap_svg(
                "60-Day Retention by Vendor-Category Breadth and Customer Segment",
                category_labels,
                segment_labels,
                breadth_segment_matrix,
                breadth_segment_counts,
            ),
        ),
        make_code_cell(
            "# 60-day retention by category breadth, split across segments\n",
            svg=grouped_bar_chart_svg(
                "60-Day Retention by Category Breadth Across Customer Segments",
                category_labels,
                [
                    ("New", category_segment_rates[0], "#b45309"),
                    ("Regular", category_segment_rates[1], "#2563eb"),
                    ("Premium", category_segment_rates[2], "#059669"),
                ],
                y_min=70,
                y_max=100,
            ),
        ),
        make_markdown_cell(
            "## Notes\n"
            "- These visuals are intended to reveal interaction patterns and guide the later formal hypothesis-testing stage.\n"
            "- No p-values, confidence intervals, or test statistics are computed here.\n"
            "- Cells with smaller sample sizes should be interpreted more carefully, which is why `n` is shown directly in the heatmaps and tables."
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
