from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "customer_level_analysis_dataset.csv"
NOTEBOOK_PATH = BASE_DIR / "notebooks" / "02_bivariate_analysis.ipynb"


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


def wrap_svg(content: str, width: int = 980, height: int = 500) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="#f8f6f2"/>'
        f"{content}</svg>"
    )


def grouped_bar_chart_svg(
    title: str,
    labels: list[str],
    series: list[tuple[str, list[float], str]],
    y_label: str = "Retention %",
    y_min: float = 0.0,
    y_max: float = 100.0,
) -> str:
    width = 980
    height = 500
    left = 90
    right = 30
    top = 70
    bottom = 130
    plot_w = width - left - right
    plot_h = height - top - bottom

    group_gap = 22
    inner_gap = 6
    group_w = (plot_w - group_gap * (len(labels) - 1)) / max(len(labels), 1)
    bar_w = (group_w - inner_gap * (len(series) - 1)) / max(len(series), 1)

    parts = [
        f'<text x="{left}" y="38" font-size="24" font-family="Helvetica, Arial, sans-serif" '
        f'font-weight="700" fill="#1f2937">{escape_xml(title)}</text>',
        f'<text x="22" y="{top + plot_h / 2}" transform="rotate(-90 22 {top + plot_h / 2})" '
        f'font-size="13" font-family="Helvetica, Arial, sans-serif" fill="#4b5563">{escape_xml(y_label)}</text>',
    ]

    for tick in range(6):
        value = y_min + (y_max - y_min) * tick / 5
        y = top + plot_h - ((value - y_min) / (y_max - y_min) * plot_h)
        parts.append(f'<line x1="{left}" y1="{y}" x2="{left + plot_w}" y2="{y}" stroke="#ddd6c8" stroke-width="1"/>')
        parts.append(
            f'<text x="{left - 12}" y="{y + 5}" text-anchor="end" font-size="12" '
            f'font-family="Helvetica, Arial, sans-serif" fill="#6b7280">{round(value, 0)}%</text>'
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
                f'<text x="{x + bar_w / 2}" y="{y - 8}" text-anchor="middle" font-size="11" '
                f'font-family="Helvetica, Arial, sans-serif" fill="#374151">{value:.1f}</text>'
            )
        parts.append(
            f'<text x="{group_x + group_w / 2}" y="{top + plot_h + 22}" text-anchor="middle" font-size="11" '
            f'font-family="Helvetica, Arial, sans-serif" fill="#374151">{escape_xml(label)}</text>'
        )

    legend_x = left
    legend_y = height - 28
    for name, _, color in series:
        parts.append(f'<rect x="{legend_x}" y="{legend_y - 10}" width="18" height="18" fill="{color}"/>')
        parts.append(
            f'<text x="{legend_x + 26}" y="{legend_y + 4}" font-size="12" '
            f'font-family="Helvetica, Arial, sans-serif" fill="#374151">{escape_xml(name)}</text>'
        )
        legend_x += 170

    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#6b7280"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#6b7280"/>')
    return wrap_svg("".join(parts), width, height)


def heatmap_svg(title: str, row_labels: list[str], col_labels: list[str], matrix: list[list[float]]) -> str:
    width = 980
    height = 520
    left = 180
    right = 50
    top = 80
    bottom = 80
    plot_w = width - left - right
    plot_h = height - top - bottom
    cols = len(col_labels)
    rows = len(row_labels)
    cell_w = plot_w / max(cols, 1)
    cell_h = plot_h / max(rows, 1)

    flat = [value for row in matrix for value in row]
    min_v = min(flat) if flat else 0
    max_v = max(flat) if flat else 100

    def color(value: float) -> str:
        if max_v == min_v:
            ratio = 0.5
        else:
            ratio = (value - min_v) / (max_v - min_v)
        r = int(245 - 120 * ratio)
        g = int(214 - 90 * ratio)
        b = int(185 - 140 * ratio)
        return f"rgb({r},{g},{b})"

    parts = [
        f'<text x="{left}" y="42" font-size="24" font-family="Helvetica, Arial, sans-serif" '
        f'font-weight="700" fill="#1f2937">{escape_xml(title)}</text>'
    ]

    for c_idx, label in enumerate(col_labels):
        x = left + c_idx * cell_w + cell_w / 2
        parts.append(
            f'<text x="{x}" y="{top - 16}" text-anchor="middle" font-size="12" '
            f'font-family="Helvetica, Arial, sans-serif" fill="#374151">{escape_xml(label)}</text>'
        )

    for r_idx, label in enumerate(row_labels):
        y = top + r_idx * cell_h + cell_h / 2 + 5
        parts.append(
            f'<text x="{left - 12}" y="{y}" text-anchor="end" font-size="12" '
            f'font-family="Helvetica, Arial, sans-serif" fill="#374151">{escape_xml(label)}</text>'
        )
        for c_idx, value in enumerate(matrix[r_idx]):
            x = left + c_idx * cell_w
            y0 = top + r_idx * cell_h
            parts.append(f'<rect x="{x}" y="{y0}" width="{cell_w}" height="{cell_h}" fill="{color(value)}" stroke="#fff" stroke-width="2"/>')
            parts.append(
                f'<text x="{x + cell_w / 2}" y="{y0 + cell_h / 2 + 5}" text-anchor="middle" font-size="12" '
                f'font-family="Helvetica, Arial, sans-serif" fill="#111827">{value:.1f}%</text>'
            )

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
                "data": {"image/svg+xml": svg, "text/plain": ["<bivariate plot>"]},
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


def retention_rate(rows: list[dict[str, str]], window: str) -> float:
    values = [row[f"returned_after_first_order_{window}d"] for row in rows if row[f"returned_after_first_order_{window}d"] != ""]
    return round(sum(int(value) for value in values) / len(values) * 100, 1) if values else 0.0


def grouped_retention(rows: list[dict[str, str]], label_fn) -> tuple[list[str], list[float], list[float], list[float], list[int]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[label_fn(row)].append(row)
    labels = list(groups.keys())
    r30 = [retention_rate(groups[label], "30") for label in labels]
    r60 = [retention_rate(groups[label], "60") for label in labels]
    r90 = [retention_rate(groups[label], "90") for label in labels]
    counts = [len(groups[label]) for label in labels]
    return labels, r30, r60, r90, counts


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


def format_group_table(title: str, labels: list[str], counts: list[int], r30: list[float], r60: list[float], r90: list[float]) -> str:
    lines = [
        f"### {title}",
        "",
        "| Group | Customers | 30d Retention % | 60d Retention % | 90d Retention % |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, count, a, b, c in zip(labels, counts, r30, r60, r90):
        lines.append(f"| {label} | {count} | {a} | {b} | {c} |")
    return "\n".join(lines)


def build_notebook(rows: list[dict[str, str]]) -> dict:
    on_time_labels = ["< 50%", "50%-69%", "70%-84%", "85%-94%", "95%-100%"]
    delay_labels = ["0-2 min", "2-5 min", "5-10 min", "10-15 min", "15+ min"]
    rating_labels = ["1.0-2.9", "3.0-3.4", "3.5-3.9", "4.0-4.4", "4.5-5.0"]
    promo_labels = ["0%", "1%-25%", "26%-50%", "51%-75%", "76%-100%"]
    breadth_labels = ["1 category", "2 categories", "3 categories", "4 categories"]
    service_labels = ["1 type", "2 types", "3 types"]

    on_time_groups = grouped_retention(rows, lambda row: bin_on_time_rate(float(row["on_time_delivery_rate"])))
    delay_groups = grouped_retention(rows, lambda row: bin_avg_delay(float(row["avg_delay_minutes"])))
    rating_groups = grouped_retention(rows, lambda row: bin_rating(float(row["avg_customer_rating"])))
    promo_groups = grouped_retention(rows, lambda row: bin_promo_rate(float(row["promo_order_rate"])))
    category_groups = grouped_retention(rows, lambda row: f'{row["unique_vendor_categories"]} category' if row["unique_vendor_categories"] == "1" else f'{row["unique_vendor_categories"]} categories')
    service_groups = grouped_retention(rows, lambda row: f'{row["unique_delivery_types"]} type' if row["unique_delivery_types"] == "1" else f'{row["unique_delivery_types"]} types')
    multi_category_groups = grouped_retention(rows, lambda row: "Multi-category users" if row["multi_category_user_flag"] == "1" else "Single-category users")
    multi_service_groups = grouped_retention(rows, lambda row: "Multi-service users" if row["multi_delivery_type_user_flag"] == "1" else "Single-service users")

    # Reorder selected groups for cleaner visuals
    def reorder(group_data, desired_labels):
        labels, r30, r60, r90, counts = group_data
        mapping = {label: (a, b, c, d) for label, a, b, c, d in zip(labels, r30, r60, r90, counts)}
        labels_out, a_out, b_out, c_out, d_out = [], [], [], [], []
        for label in desired_labels:
            if label in mapping:
                a, b, c, d = mapping[label]
                labels_out.append(label)
                a_out.append(a)
                b_out.append(b)
                c_out.append(c)
                d_out.append(d)
        return labels_out, a_out, b_out, c_out, d_out

    on_time_groups = reorder(on_time_groups, on_time_labels)
    delay_groups = reorder(delay_groups, delay_labels)
    rating_groups = reorder(rating_groups, rating_labels)
    promo_groups = reorder(promo_groups, promo_labels)
    category_groups = reorder(category_groups, breadth_labels)
    service_groups = reorder(service_groups, service_labels)
    multi_category_groups = reorder(multi_category_groups, ["Single-category users", "Multi-category users"])
    multi_service_groups = reorder(multi_service_groups, ["Single-service users", "Multi-service users"])

    heatmap_rows = ["Single-category users", "Multi-category users"]
    heatmap_cols = ["Single-service users", "Multi-service users"]
    heatmap_matrix = []
    for category_label in heatmap_rows:
        heatmap_row = []
        for service_label in heatmap_cols:
            subset = [
                row for row in rows
                if ("Multi-category users" if row["multi_category_user_flag"] == "1" else "Single-category users") == category_label
                and ("Multi-service users" if row["multi_delivery_type_user_flag"] == "1" else "Single-service users") == service_label
            ]
            heatmap_row.append(retention_rate(subset, "60"))
        heatmap_matrix.append(heatmap_row)

    cells = [
        make_markdown_cell(
            "# 02 Bivariate Analysis\n"
            "This notebook visualizes relationships between engineered features and retention outcomes for the three project hypotheses. "
            "It is intentionally descriptive only and does not perform any statistical hypothesis testing."
        ),
        make_markdown_cell(
            "## Scope\n"
            "- Hypothesis 1: delivery reliability and retention\n"
            "- Hypothesis 2: promotion behavior and retention\n"
            "- Hypothesis 3: cross-category / cross-service usage and retention\n"
            "- Retention is shown using observed `30-day`, `60-day`, and `90-day` return rates"
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
        make_markdown_cell(format_group_table("Retention by On-Time Delivery Rate Bucket", *on_time_groups)),
        make_code_cell(
            "# Delivery reliability: on-time rate bucket vs retention\n",
            svg=grouped_bar_chart_svg(
                "Retention by On-Time Delivery Rate Bucket",
                on_time_groups[0],
                [
                    ("30-day", on_time_groups[1], "#b45309"),
                    ("60-day", on_time_groups[2], "#2563eb"),
                    ("90-day", on_time_groups[3], "#059669"),
                ],
                y_min=60,
                y_max=100,
            ),
        ),
        make_markdown_cell(format_group_table("Retention by Average Delay Bucket", *delay_groups)),
        make_code_cell(
            "# Delivery reliability: average delay bucket vs retention\n",
            svg=grouped_bar_chart_svg(
                "Retention by Average Delay Bucket",
                delay_groups[0],
                [
                    ("30-day", delay_groups[1], "#c2410c"),
                    ("60-day", delay_groups[2], "#2563eb"),
                    ("90-day", delay_groups[3], "#059669"),
                ],
                y_min=60,
                y_max=100,
            ),
        ),
        make_markdown_cell(format_group_table("Retention by Average Customer Rating Bucket", *rating_groups)),
        make_code_cell(
            "# Delivery satisfaction: average rating bucket vs retention\n",
            svg=grouped_bar_chart_svg(
                "Retention by Average Customer Rating Bucket",
                rating_groups[0],
                [
                    ("30-day", rating_groups[1], "#be185d"),
                    ("60-day", rating_groups[2], "#2563eb"),
                    ("90-day", rating_groups[3], "#059669"),
                ],
                y_min=60,
                y_max=100,
            ),
        ),
        make_markdown_cell("## Hypothesis 2: Promotion Behavior vs Retention"),
        make_markdown_cell(format_group_table("Retention by Promotion Usage Rate Bucket", *promo_groups)),
        make_code_cell(
            "# Promotion intensity vs retention\n",
            svg=grouped_bar_chart_svg(
                "Retention by Promotion Usage Rate Bucket",
                promo_groups[0],
                [
                    ("30-day", promo_groups[1], "#7c3aed"),
                    ("60-day", promo_groups[2], "#2563eb"),
                    ("90-day", promo_groups[3], "#059669"),
                ],
                y_min=60,
                y_max=100,
            ),
        ),
        make_markdown_cell("## Hypothesis 3: Breadth of Platform Usage vs Retention"),
        make_markdown_cell(format_group_table("Retention by Number of Vendor Categories Used", *category_groups)),
        make_code_cell(
            "# Category breadth vs retention\n",
            svg=grouped_bar_chart_svg(
                "Retention by Number of Vendor Categories Used",
                category_groups[0],
                [
                    ("30-day", category_groups[1], "#0f766e"),
                    ("60-day", category_groups[2], "#2563eb"),
                    ("90-day", category_groups[3], "#059669"),
                ],
                y_min=60,
                y_max=100,
            ),
        ),
        make_markdown_cell(format_group_table("Retention by Number of Delivery Types Used", *service_groups)),
        make_code_cell(
            "# Service breadth vs retention\n",
            svg=grouped_bar_chart_svg(
                "Retention by Number of Delivery Types Used",
                service_groups[0],
                [
                    ("30-day", service_groups[1], "#1d4ed8"),
                    ("60-day", service_groups[2], "#0891b2"),
                    ("90-day", service_groups[3], "#059669"),
                ],
                y_min=60,
                y_max=100,
            ),
        ),
        make_markdown_cell(format_group_table("Retention by Single vs Multi-Category Usage", *multi_category_groups)),
        make_code_cell(
            "# Single vs multi-category users\n",
            svg=grouped_bar_chart_svg(
                "Retention by Single vs Multi-Category Usage",
                multi_category_groups[0],
                [
                    ("30-day", multi_category_groups[1], "#b45309"),
                    ("60-day", multi_category_groups[2], "#2563eb"),
                    ("90-day", multi_category_groups[3], "#059669"),
                ],
                y_min=60,
                y_max=100,
            ),
        ),
        make_markdown_cell(format_group_table("Retention by Single vs Multi-Service Usage", *multi_service_groups)),
        make_code_cell(
            "# Single vs multi-service users\n",
            svg=grouped_bar_chart_svg(
                "Retention by Single vs Multi-Service Usage",
                multi_service_groups[0],
                [
                    ("30-day", multi_service_groups[1], "#b45309"),
                    ("60-day", multi_service_groups[2], "#2563eb"),
                    ("90-day", multi_service_groups[3], "#059669"),
                ],
                y_min=60,
                y_max=100,
            ),
        ),
        make_code_cell(
            "# Combined view: category breadth x service breadth, shown with 60-day retention only\n",
            svg=heatmap_svg(
                "60-Day Retention by Category and Service Breadth",
                heatmap_rows,
                heatmap_cols,
                heatmap_matrix,
            ),
        ),
        make_markdown_cell(
            "## Notes\n"
            "- These visuals are descriptive and intended to guide later formal testing.\n"
            "- Because the environment lacks plotting libraries, the notebook uses embedded SVG charts for quick review.\n"
            "- Recent customers with incomplete observation windows are excluded from the relevant retention-rate denominators."
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
