from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
INTERIM_DIR = BASE_DIR / "data" / "interim"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"
OUTPUT_PATH = REPORTS_DIR / "urban_cart_retention_report.html"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def percentage(value: float) -> str:
    return f"{value * 100:.1f}%"


def fmt_int(value: int) -> str:
    return f"{value:,}"


def svg_bar_chart(
    title: str,
    subtitle: str,
    labels: list[str],
    values: list[float],
    *,
    percent: bool = False,
    fill: str = "#0f766e",
    width: int = 560,
    height: int = 320,
) -> str:
    max_value = max(values) if values else 1
    chart_left = 52
    chart_bottom = height - 42
    chart_top = 26
    chart_width = width - chart_left - 24
    chart_height = chart_bottom - chart_top
    bar_gap = 18
    bar_width = max(18, (chart_width - bar_gap * (len(values) - 1)) / max(len(values), 1))

    bars = []
    grid = []
    for step in range(5):
        y = chart_bottom - chart_height * (step / 4)
        grid.append(
            f'<line x1="{chart_left}" y1="{y:.1f}" x2="{width - 18}" y2="{y:.1f}" '
            f'stroke="rgba(15,23,42,0.08)" stroke-width="1" />'
        )

    for index, (label, value) in enumerate(zip(labels, values)):
        x = chart_left + index * (bar_width + bar_gap)
        bar_height = 0 if max_value == 0 else chart_height * (value / max_value)
        y = chart_bottom - bar_height
        display = f"{value * 100:.1f}%" if percent else f"{value:.0f}"
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" '
            f'rx="10" fill="{fill}" />'
        )
        bars.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{y - 10:.1f}" text-anchor="middle" '
            f'font-size="12" fill="#0f172a">{display}</text>'
        )
        bars.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{chart_bottom + 20:.1f}" text-anchor="middle" '
            f'font-size="12" fill="#475569">{label}</text>'
        )

    return (
        f'<figure class="chart-card">'
        f'<figcaption><h3>{title}</h3><p>{subtitle}</p></figcaption>'
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{title}">'
        f'{"".join(grid)}'
        f'<line x1="{chart_left}" y1="{chart_bottom}" x2="{width - 18}" y2="{chart_bottom}" stroke="#94a3b8" stroke-width="1.5" />'
        f'{"".join(bars)}'
        f"</svg></figure>"
    )


def svg_lollipop_chart(
    title: str,
    subtitle: str,
    labels: list[str],
    values: list[float],
    *,
    width: int = 560,
    height: int = 320,
    accent: str = "#c2410c",
) -> str:
    chart_left = 94
    chart_right = width - 30
    chart_top = 34
    chart_bottom = height - 34
    row_gap = (chart_bottom - chart_top) / max(len(labels), 1)
    min_value = min(values)
    max_value = max(values)
    domain_start = math.floor(min_value * 100 / 5) * 5
    domain_end = math.ceil(max_value * 100 / 5) * 5
    if domain_end == domain_start:
        domain_end += 5

    elements = []
    for tick in range(domain_start, domain_end + 1, 5):
        x = chart_left + (chart_right - chart_left) * ((tick - domain_start) / (domain_end - domain_start))
        elements.append(
            f'<line x1="{x:.1f}" y1="{chart_top - 14}" x2="{x:.1f}" y2="{chart_bottom}" '
            f'stroke="rgba(15,23,42,0.08)" stroke-width="1" />'
        )
        elements.append(
            f'<text x="{x:.1f}" y="{chart_top - 18:.1f}" text-anchor="middle" font-size="12" fill="#64748b">{tick}%</text>'
        )

    for idx, (label, value) in enumerate(zip(labels, values)):
        y = chart_top + row_gap * idx + row_gap / 2
        x = chart_left + (chart_right - chart_left) * (((value * 100) - domain_start) / (domain_end - domain_start))
        elements.append(
            f'<text x="{chart_left - 12}" y="{y + 4:.1f}" text-anchor="end" font-size="12" fill="#475569">{label}</text>'
        )
        elements.append(
            f'<line x1="{chart_left}" y1="{y:.1f}" x2="{x:.1f}" y2="{y:.1f}" stroke="{accent}" stroke-width="4" stroke-linecap="round" opacity="0.35" />'
        )
        elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{accent}" />')
        elements.append(
            f'<text x="{x + 12:.1f}" y="{y + 4:.1f}" font-size="12" fill="#0f172a">{value * 100:.1f}%</text>'
        )

    return (
        f'<figure class="chart-card">'
        f'<figcaption><h3>{title}</h3><p>{subtitle}</p></figcaption>'
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{title}">'
        f'{"".join(elements)}'
        f"</svg></figure>"
    )


def svg_stacked_bar_chart(
    title: str,
    subtitle: str,
    labels: list[str],
    series: list[tuple[str, str, list[float]]],
    *,
    width: int = 560,
    height: int = 320,
) -> str:
    chart_left = 64
    chart_bottom = height - 52
    chart_top = 24
    chart_width = width - chart_left - 24
    chart_height = chart_bottom - chart_top
    group_gap = 28
    bar_width = max(28, (chart_width - group_gap * (len(labels) - 1)) / max(len(labels), 1))

    parts = []
    for step in range(5):
        y = chart_bottom - chart_height * (step / 4)
        parts.append(
            f'<line x1="{chart_left}" y1="{y:.1f}" x2="{width - 16}" y2="{y:.1f}" stroke="rgba(15,23,42,0.08)" stroke-width="1" />'
        )
        parts.append(
            f'<text x="{chart_left - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="11" fill="#64748b">{int(step * 25)}%</text>'
        )

    for idx, label in enumerate(labels):
        x = chart_left + idx * (bar_width + group_gap)
        current_bottom = chart_bottom
        for _, color, values in series:
            value = values[idx]
            bar_height = chart_height * value
            y = current_bottom - bar_height
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" rx="8" fill="{color}" />'
            )
            current_bottom = y
        parts.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{chart_bottom + 18:.1f}" text-anchor="middle" font-size="12" fill="#475569">{label}</text>'
        )

    legend = []
    legend_x = chart_left
    for name, color, _ in series:
        legend.append(f'<rect x="{legend_x}" y="{height - 22}" width="12" height="12" rx="3" fill="{color}" />')
        legend.append(f'<text x="{legend_x + 18}" y="{height - 12}" font-size="12" fill="#475569">{name}</text>')
        legend_x += 112

    return (
        f'<figure class="chart-card">'
        f'<figcaption><h3>{title}</h3><p>{subtitle}</p></figcaption>'
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{title}">'
        f'{"".join(parts)}{"".join(legend)}'
        f"</svg></figure>"
    )


def svg_hbar_chart(
    title: str,
    subtitle: str,
    labels: list[str],
    values: list[int],
    *,
    width: int = 560,
    height: int = 360,
    fill: str = "#1d4ed8",
) -> str:
    chart_left = 116
    chart_right = width - 24
    chart_top = 24
    chart_bottom = height - 28
    row_gap = (chart_bottom - chart_top) / max(len(labels), 1)
    max_value = max(values) if values else 1
    parts = []

    for idx, (label, value) in enumerate(zip(labels, values)):
        y = chart_top + row_gap * idx + row_gap / 2
        bar_length = (chart_right - chart_left) * (value / max_value)
        parts.append(
            f'<text x="{chart_left - 12}" y="{y + 4:.1f}" text-anchor="end" font-size="12" fill="#475569">{label}</text>'
        )
        parts.append(
            f'<rect x="{chart_left}" y="{y - 10:.1f}" width="{bar_length:.1f}" height="20" rx="10" fill="{fill}" opacity="{0.92 - idx * 0.06:.2f}" />'
        )
        parts.append(
            f'<text x="{chart_left + bar_length + 10:.1f}" y="{y + 4:.1f}" font-size="12" fill="#0f172a">{value}</text>'
        )

    return (
        f'<figure class="chart-card">'
        f'<figcaption><h3>{title}</h3><p>{subtitle}</p></figcaption>'
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{title}">'
        f'{"".join(parts)}'
        f"</svg></figure>"
    )


def svg_donut_chart(
    title: str,
    subtitle: str,
    labels: list[str],
    values: list[int],
    colors: list[str],
    *,
    width: int = 560,
    height: int = 320,
) -> str:
    total = sum(values)
    center_x = 150
    center_y = 166
    radius = 84
    stroke_width = 34
    circumference = 2 * math.pi * radius
    offset = 0.0
    segments = []
    for value, color in zip(values, colors):
        share = value / total if total else 0
        dash = circumference * share
        segments.append(
            f'<circle cx="{center_x}" cy="{center_y}" r="{radius}" fill="none" stroke="{color}" stroke-width="{stroke_width}" '
            f'stroke-dasharray="{dash:.2f} {circumference - dash:.2f}" stroke-dashoffset="{-offset:.2f}" transform="rotate(-90 {center_x} {center_y})" stroke-linecap="butt" />'
        )
        offset += dash

    legend = []
    legend_x = 300
    legend_y = 92
    for idx, (label, value, color) in enumerate(zip(labels, values, colors)):
        y = legend_y + idx * 50
        legend.append(f'<rect x="{legend_x}" y="{y}" width="14" height="14" rx="4" fill="{color}" />')
        legend.append(f'<text x="{legend_x + 22}" y="{y + 11}" font-size="13" fill="#0f172a">{label}</text>')
        legend.append(
            f'<text x="{legend_x + 22}" y="{y + 28}" font-size="12" fill="#64748b">{fmt_int(value)} customers ({percentage(value / total)})</text>'
        )

    return (
        f'<figure class="chart-card">'
        f'<figcaption><h3>{title}</h3><p>{subtitle}</p></figcaption>'
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{title}">'
        f'<circle cx="{center_x}" cy="{center_y}" r="{radius}" fill="none" stroke="rgba(148,163,184,0.18)" stroke-width="{stroke_width}" />'
        f'{"".join(segments)}'
        f'<text x="{center_x}" y="{center_y - 6}" text-anchor="middle" font-size="28" fill="#0f172a" font-weight="700">{fmt_int(total)}</text>'
        f'<text x="{center_x}" y="{center_y + 18}" text-anchor="middle" font-size="12" fill="#64748b">customers</text>'
        f'{"".join(legend)}'
        f"</svg></figure>"
    )


def build_report() -> str:
    customers = read_csv(RAW_DIR / "customers.csv")
    orders = read_csv(RAW_DIR / "orders.csv")
    deliveries = read_csv(RAW_DIR / "deliveries.csv")
    customer_level = read_csv(PROCESSED_DIR / "customer_level_analysis_dataset.csv")
    order_level = read_csv(INTERIM_DIR / "order_level_analysis_dataset.csv")

    first_orders = [row for row in order_level if row["customer_order_number"] == "1"]
    customer_by_id = {row["customer_id"]: row for row in customer_level}

    age_counts = Counter(row["age_group"] for row in customers)
    device_counts = Counter(row["preferred_device"] for row in customers)
    top_cities = Counter(row["city"] for row in customers).most_common(8)
    repeat_rate = sum(int(row["repeat_customer_flag"]) for row in customer_level) / len(customer_level)

    retention_windows = []
    for window in [30, 60, 90]:
        flag = f"full_{window}d_observation_flag"
        outcome = f"returned_after_first_order_{window}d"
        eligible = [row for row in customer_level if row[flag] == "1" and row[outcome] != ""]
        retained = sum(int(row[outcome]) for row in eligible)
        retention_windows.append((window, len(eligible), retained / len(eligible)))

    first_order_30 = []
    for row in first_orders:
        customer = customer_by_id[row["customer_id"]]
        if customer["full_30d_observation_flag"] == "1" and customer["returned_after_first_order_30d"] != "":
            first_order_30.append(
                {
                    "delivery_status": row["delivery_status"],
                    "order_accuracy": row["order_accuracy"],
                    "promo_used": row["promo_used"],
                    "customer_rating": int(row["customer_rating"]),
                    "delay_minutes": int(row["delay_minutes"]),
                    "returned": int(customer["returned_after_first_order_30d"]),
                }
            )

    def retention_by(group_key: str) -> dict[str, float]:
        grouped: dict[str, list[int]] = defaultdict(list)
        for row in first_order_30:
            grouped[str(row[group_key])].append(row["returned"])
        return {key: sum(values) / len(values) for key, values in grouped.items()}

    ratings = retention_by("customer_rating")
    delivery_status = retention_by("delivery_status")
    promo = retention_by("promo_used")
    accuracy = retention_by("order_accuracy")

    delay_groups: dict[str, list[int]] = defaultdict(list)
    for row in first_order_30:
        delay = row["delay_minutes"]
        if delay == 0:
            bucket = "0 min"
        elif delay <= 5:
            bucket = "1-5"
        elif delay <= 15:
            bucket = "6-15"
        else:
            bucket = "15+"
        delay_groups[bucket].append(row["returned"])
    delay_bucket_rates = {
        label: sum(values) / len(values)
        for label, values in delay_groups.items()
    }

    hero_stats = [
        ("Customers", fmt_int(len(customers))),
        ("Orders", fmt_int(len(orders))),
        ("Cities", fmt_int(len(set(row["city"] for row in customers)))),
        ("Repeat customer rate", percentage(repeat_rate)),
    ]

    retention_chart = svg_bar_chart(
        "Retention Strength Across Windows",
        "The business is strongest after the first return, but 30 days remains the most revealing risk window.",
        [f"{window}-day" for window, _, _ in retention_windows],
        [rate for _, _, rate in retention_windows],
        percent=True,
        fill="#0f766e",
    )
    device_chart = svg_donut_chart(
        "Channel Mix",
        "Mobile app usage dominates the customer base, which makes it the center of the retention experience.",
        ["Mobile App", "Web"],
        [device_counts["Mobile App"], device_counts["Web"]],
        ["#0f766e", "#fb923c"],
    )
    city_chart = svg_hbar_chart(
        "Largest Customer Cities",
        "UrbanCart demand is spread across multiple cities rather than concentrated in a single market.",
        [city for city, _ in top_cities],
        [count for _, count in top_cities],
        fill="#1d4ed8",
    )
    rating_chart = svg_lollipop_chart(
        "30-Day Retention by First-Order Rating",
        "Customer satisfaction creates the clearest step-up in return behavior.",
        [str(key) for key in sorted(ratings)],
        [ratings[key] for key in sorted(ratings)],
        accent="#0f766e",
    )
    delivery_chart = svg_bar_chart(
        "30-Day Retention by Delivery Status",
        "Delayed first orders have a visible retention penalty even before adjustment.",
        ["Delayed", "On-Time"],
        [delivery_status["Delayed"], delivery_status["On-Time"]],
        percent=True,
        fill="#c2410c",
    )
    promo_chart = svg_bar_chart(
        "30-Day Retention by Promo Usage",
        "Promotions show only a small raw difference and no reliable lift after adjustment.",
        ["No Promo", "Promo Used"],
        [promo["No"], promo["Yes"]],
        percent=True,
        fill="#7c3aed",
    )
    delay_chart = svg_lollipop_chart(
        "30-Day Retention by Delay Bucket",
        "The steepest drop appears once delays reach 15 minutes or more.",
        ["0 min", "1-5", "6-15", "15+"],
        [
            delay_bucket_rates.get("0 min", 0.0),
            delay_bucket_rates.get("1-5", 0.0),
            delay_bucket_rates.get("6-15", 0.0),
            delay_bucket_rates.get("15+", 0.0),
        ],
        accent="#c2410c",
    )
    age_chart = svg_hbar_chart(
        "Age Group Composition",
        "The largest customer cluster sits in the 26-35 range, followed by 36-45.",
        ["26-35", "36-45", "18-25", "46-55", "56+"],
        [age_counts["26-35"], age_counts["36-45"], age_counts["18-25"], age_counts["46-55"], age_counts["56+"]],
        fill="#0f766e",
    )
    stacked_chart = svg_stacked_bar_chart(
        "First-Order Accuracy Mix",
        "Most orders are accurate, but the small share of wrong or partial orders still matters when customers are evaluating the service.",
        ["30-day cohort"],
        [
            ("Correct", "#0f766e", [4448 / 4855]),
            ("Partial", "#f59e0b", [304 / 4855]),
            ("Wrong", "#dc2626", [103 / 4855]),
        ],
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>UrbanCart Retention Analysis Report</title>
  <style>
    :root {{
      --ink: #0f172a;
      --muted: #475569;
      --soft: #f8fafc;
      --paper: #fffdf8;
      --line: rgba(15, 23, 42, 0.09);
      --teal: #0f766e;
      --teal-soft: #ccfbf1;
      --orange: #c2410c;
      --orange-soft: #ffedd5;
      --blue: #1d4ed8;
      --blue-soft: #dbeafe;
      --violet: #7c3aed;
      --violet-soft: #ede9fe;
      --shadow: 0 24px 60px rgba(15, 23, 42, 0.08);
      --radius: 22px;
    }}

    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      font-family: "Avenir Next", "Trebuchet MS", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(15,118,110,0.12), transparent 28%),
        radial-gradient(circle at top right, rgba(194,65,12,0.10), transparent 24%),
        linear-gradient(180deg, #fffdf8 0%, #fffaf3 38%, #f8fafc 100%);
      line-height: 1.6;
    }}

    a {{ color: inherit; }}

    .page {{
      width: min(1180px, calc(100vw - 40px));
      margin: 0 auto;
      padding: 28px 0 64px;
    }}

    .hero {{
      position: relative;
      overflow: hidden;
      padding: 42px;
      border: 1px solid rgba(15, 23, 42, 0.06);
      border-radius: 30px;
      background:
        linear-gradient(135deg, rgba(255,255,255,0.94), rgba(255,248,240,0.95)),
        linear-gradient(135deg, rgba(15,118,110,0.06), rgba(194,65,12,0.05));
      box-shadow: var(--shadow);
    }}

    .hero::after {{
      content: "";
      position: absolute;
      inset: auto -40px -60px auto;
      width: 260px;
      height: 260px;
      background: radial-gradient(circle, rgba(15,118,110,0.18), transparent 68%);
      pointer-events: none;
    }}

    .eyebrow {{
      display: inline-flex;
      align-items: center;
      gap: 10px;
      padding: 8px 14px;
      border-radius: 999px;
      background: rgba(15,118,110,0.09);
      color: var(--teal);
      font-size: 13px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      font-weight: 700;
    }}

    h1, h2, h3, h4 {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
      line-height: 1.1;
      letter-spacing: -0.02em;
    }}

    .hero h1 {{
      max-width: 760px;
      margin-top: 18px;
      font-size: clamp(2.6rem, 5vw, 4.5rem);
    }}

    .hero p {{
      max-width: 760px;
      margin: 18px 0 0;
      font-size: 1.08rem;
      color: var(--muted);
    }}

    .hero-grid {{
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 22px;
      margin-top: 34px;
    }}

    .summary-panel {{
      border-radius: 24px;
      padding: 20px 22px;
      background: rgba(255,255,255,0.72);
      border: 1px solid rgba(15,23,42,0.06);
      backdrop-filter: blur(8px);
    }}

    .summary-panel strong {{
      color: var(--teal);
      font-size: 0.94rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}

    .hero-stats {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }}

    .stat-card {{
      padding: 18px;
      border-radius: 20px;
      background: rgba(255,255,255,0.84);
      border: 1px solid rgba(15,23,42,0.06);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.7);
    }}

    .stat-label {{
      display: block;
      color: var(--muted);
      font-size: 0.92rem;
      margin-bottom: 6px;
    }}

    .stat-value {{
      font-size: 1.9rem;
      font-weight: 700;
      line-height: 1;
    }}

    nav {{
      position: sticky;
      top: 10px;
      z-index: 5;
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin: 22px 0 26px;
      padding: 12px;
      border-radius: 18px;
      background: rgba(255,255,255,0.8);
      border: 1px solid rgba(15,23,42,0.06);
      backdrop-filter: blur(8px);
      box-shadow: 0 10px 25px rgba(15,23,42,0.05);
    }}

    nav a {{
      padding: 10px 14px;
      border-radius: 999px;
      text-decoration: none;
      color: var(--muted);
      background: rgba(248,250,252,0.96);
      border: 1px solid rgba(15,23,42,0.05);
      font-size: 0.95rem;
    }}

    section {{
      margin-top: 24px;
      padding: 30px;
      border-radius: 28px;
      background: rgba(255,255,255,0.82);
      border: 1px solid rgba(15,23,42,0.06);
      box-shadow: var(--shadow);
    }}

    .section-head {{
      display: flex;
      flex-wrap: wrap;
      align-items: end;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 18px;
    }}

    .section-head p {{
      margin: 0;
      max-width: 720px;
      color: var(--muted);
    }}

    .grid-2 {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 22px;
    }}

    .grid-3 {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 18px;
    }}

    .copy-card {{
      padding: 20px 22px;
      border-radius: 22px;
      background: linear-gradient(180deg, rgba(248,250,252,0.88), rgba(255,255,255,0.92));
      border: 1px solid rgba(15,23,42,0.06);
    }}

    .copy-card p {{
      margin: 0 0 12px;
      color: var(--muted);
    }}

    .copy-card p:last-child {{ margin-bottom: 0; }}

    .accent-teal {{ border-left: 5px solid var(--teal); }}
    .accent-orange {{ border-left: 5px solid var(--orange); }}
    .accent-blue {{ border-left: 5px solid var(--blue); }}
    .accent-violet {{ border-left: 5px solid var(--violet); }}

    .chart-card {{
      margin: 0;
      padding: 18px 18px 10px;
      border-radius: 24px;
      background: #fff;
      border: 1px solid rgba(15,23,42,0.06);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.75);
    }}

    .chart-card figcaption {{
      margin-bottom: 10px;
    }}

    .chart-card h3 {{
      font-size: 1.18rem;
      margin-bottom: 6px;
    }}

    .chart-card p {{
      margin: 0;
      color: var(--muted);
      font-size: 0.96rem;
    }}

    .chart-card svg {{
      width: 100%;
      height: auto;
      display: block;
    }}

    .pill-list {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 10px;
    }}

    .pill {{
      padding: 9px 12px;
      border-radius: 999px;
      font-size: 0.92rem;
      background: #f8fafc;
      border: 1px solid rgba(15,23,42,0.06);
      color: var(--muted);
    }}

    .findings {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
      margin-top: 16px;
    }}

    .finding {{
      padding: 20px;
      border-radius: 22px;
      background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,250,252,0.9));
      border: 1px solid rgba(15,23,42,0.06);
    }}

    .finding strong {{
      display: inline-block;
      margin-bottom: 8px;
      color: var(--ink);
    }}

    .recommendation-list,
    .insight-list {{
      margin: 14px 0 0;
      padding-left: 18px;
      color: var(--muted);
    }}

    .recommendation-list li,
    .insight-list li {{
      margin-bottom: 10px;
    }}

    footer {{
      margin-top: 22px;
      padding: 24px 28px;
      border-radius: 24px;
      background: linear-gradient(135deg, rgba(15,118,110,0.09), rgba(29,78,216,0.08));
      border: 1px solid rgba(15,23,42,0.06);
      color: var(--muted);
    }}

    @media (max-width: 920px) {{
      .hero-grid,
      .grid-2,
      .grid-3,
      .findings {{
        grid-template-columns: 1fr;
      }}

      .page {{
        width: min(100vw, calc(100vw - 24px));
      }}

      section,
      .hero {{
        padding: 22px;
      }}

      nav {{
        position: static;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <header class="hero">
      <span class="eyebrow">UrbanCart Customer Retention Study</span>
      <h1>What keeps customers coming back after their first order?</h1>
      <p>
        This report combines exploratory analysis and formal hypothesis testing to identify the operating
        signals most closely linked to customer retention at UrbanCart. The emphasis is practical: which
        experiences appear to build repeat behavior, and which levers look weaker than expected.
      </p>
      <div class="hero-grid">
        <div class="summary-panel">
          <strong>Executive takeaway</strong>
          <p>
            Retention at UrbanCart is strong overall, but the first-order experience still matters. The clearest
            signal in both the descriptive analysis and the hypothesis tests is customer satisfaction: higher
            first-order ratings consistently align with higher return rates. Promotions, by contrast, do not
            show a statistically reliable retention lift.
          </p>
        </div>
        <div class="hero-stats">
          {"".join(f'<div class="stat-card"><span class="stat-label">{label}</span><div class="stat-value">{value}</div></div>' for label, value in hero_stats)}
        </div>
      </div>
    </header>

    <nav>
      <a href="#problem">Problem</a>
      <a href="#data">Data Overview</a>
      <a href="#patterns">Descriptive Analysis</a>
      <a href="#testing">Hypothesis Testing</a>
      <a href="#insights">Key Insights</a>
      <a href="#recommendations">Recommendations</a>
      <a href="#limits">Limitations</a>
    </nav>

    <section id="problem">
      <div class="section-head">
        <div>
          <h2>Problem Statement</h2>
          <p>The business question is not whether retention matters, but which parts of the customer experience are shaping it most.</p>
        </div>
      </div>
      <div class="grid-2">
        <div class="copy-card accent-teal">
          <p>
            UrbanCart operates in a highly competitive delivery market where retaining existing customers is
            more efficient than constantly replacing them with new acquisitions. The project goal was to identify
            the factors most associated with continued activity and repeat ordering.
          </p>
          <p>
            Three hypotheses were originally proposed around delivery reliability, promotion-driven engagement,
            and breadth of platform usage. Based on your latest direction, this report formally tests the first
            two and leaves the third outside the final inferential scope.
          </p>
        </div>
        <div class="copy-card accent-orange">
          <p><strong>Primary KPIs</strong></p>
          <div class="pill-list">
            <span class="pill">Customer retention rate</span>
            <span class="pill">Repeat order rate</span>
            <span class="pill">Average order frequency</span>
          </div>
          <p style="margin-top:14px;">
            For the formal tests, retention is defined as whether a customer returns within 30, 60, or 90 days
            after the first order, using only customers with a full observation window for each metric.
          </p>
        </div>
      </div>
    </section>

    <section id="data">
      <div class="section-head">
        <div>
          <h2>Data Overview</h2>
          <p>The available data is broad enough for strong retention analysis, especially on operations and promotion usage.</p>
        </div>
      </div>
      <div class="grid-3">
        <div class="copy-card accent-blue">
          <p><strong>Raw sources</strong></p>
          <p>`customers.csv`, `orders.csv`, and `deliveries.csv` form a complete customer-order-delivery chain with no reported missing values in the project review.</p>
        </div>
        <div class="copy-card accent-teal">
          <p><strong>Coverage</strong></p>
          <p>{fmt_int(len(customers))} customers generated {fmt_int(len(orders))} orders and {fmt_int(len(deliveries))} delivery records across 12 cities and 8 states.</p>
        </div>
        <div class="copy-card accent-orange">
          <p><strong>Behavioral shape</strong></p>
          <p>The customer base is mobile-first, geographically distributed, and heavily repeat-oriented, which makes it a strong setting for retention analysis.</p>
        </div>
      </div>
      <div class="grid-2" style="margin-top:22px;">
        {device_chart}
        {city_chart}
      </div>
      <div class="grid-2" style="margin-top:22px;">
        {age_chart}
        {retention_chart}
      </div>
    </section>

    <section id="patterns">
      <div class="section-head">
        <div>
          <h2>Descriptive Analysis</h2>
          <p>The descriptive layer pointed to one dominant storyline: early experience quality appears more important than promotional intensity.</p>
        </div>
      </div>
      <div class="findings">
        <div class="finding">
          <strong>Delivery quality stood out immediately.</strong>
          <p>
            Customers with cleaner delivery experiences were more likely to come back. In the 30-day eligible cohort,
            on-time first orders retained at <strong>{percentage(delivery_status["On-Time"])}</strong> versus
            <strong>{percentage(delivery_status["Delayed"])}</strong> for delayed orders.
          </p>
        </div>
        <div class="finding">
          <strong>Satisfaction moved almost step-for-step with retention.</strong>
          <p>
            First-order ratings rose alongside return rates, from <strong>{percentage(ratings["1"])}</strong> for rating 1
            to <strong>{percentage(ratings["5"])}</strong> for rating 5. That makes customer rating the cleanest descriptive
            indicator in the project.
          </p>
        </div>
        <div class="finding">
          <strong>Promotions looked mixed rather than powerful.</strong>
          <p>
            30-day retention was <strong>{percentage(promo["Yes"])}</strong> for promo users and
            <strong>{percentage(promo["No"])}</strong> for non-promo users. The raw difference exists, but it is small.
          </p>
        </div>
        <div class="finding">
          <strong>The early window is the most sensitive one.</strong>
          <p>
            Retention rises from <strong>{percentage(retention_windows[0][2])}</strong> at 30 days to
            <strong>{percentage(retention_windows[2][2])}</strong> at 90 days, which suggests that customers who return
            once often continue to stay active.
          </p>
        </div>
      </div>
      <div class="grid-2" style="margin-top:22px;">
        {rating_chart}
        {delivery_chart}
      </div>
      <div class="grid-2" style="margin-top:22px;">
        {delay_chart}
        {stacked_chart}
      </div>
    </section>

    <section id="testing">
      <div class="section-head">
        <div>
          <h2>Hypothesis Testing Results</h2>
          <p>Formal testing was run on the first two approved hypotheses using first-order cohorts and 30, 60, and 90-day retention windows.</p>
        </div>
      </div>
      <div class="grid-2">
        <div class="copy-card accent-teal">
          <h3 style="margin-bottom:12px;">Hypothesis 1: Delivery reliability</h3>
          <p>
            Supported in direction. The strongest result came from customer rating, which remained statistically
            significant after adjustment in both the 30-day and 60-day logistic models.
          </p>
          <ul class="insight-list">
            <li>30-day model: customer rating OR = 1.357, 95% CI [1.252, 1.470], p &lt; 0.0001.</li>
            <li>60-day model: customer rating OR = 1.459, 95% CI [1.283, 1.658], p &lt; 0.0001.</li>
            <li>Delayed first orders had lower raw retention, and delivery-status chi-square tests were significant across all three windows.</li>
            <li>Delay minutes alone were not a stable adjusted predictor once the broader model controls were added.</li>
          </ul>
        </div>
        <div class="copy-card accent-violet">
          <h3 style="margin-bottom:12px;">Hypothesis 2: Promotion-driven engagement</h3>
          <p>
            Not supported. Promotions did not deliver a statistically reliable retention lift in either the
            bivariate tests or the adjusted first-order logistic models.
          </p>
          <ul class="insight-list">
            <li>30-day model: promo-used OR = 1.062, 95% CI [0.908, 1.241], p = 0.4511.</li>
            <li>60-day model: promo-used OR = 1.217, 95% CI [0.935, 1.585], p = 0.1443.</li>
            <li>Discount depth among promo users also failed to show a reliable positive effect.</li>
            <li>Personalization remains untestable because the data does not include campaign, recommendation, or CRM exposure fields.</li>
          </ul>
        </div>
      </div>
      <div class="grid-2" style="margin-top:22px;">
        {promo_chart}
        {retention_chart}
      </div>
    </section>

    <section id="insights">
      <div class="section-head">
        <div>
          <h2>Key Insights</h2>
          <p>The combined exploratory and inferential view converges on a clear set of operational lessons.</p>
        </div>
      </div>
      <div class="grid-3">
        <div class="copy-card accent-teal">
          <p><strong>1. The first order is a retention test.</strong></p>
          <p>
            The 30-day window carries the most contrast, which means early customer experience management is likely
            more valuable than waiting to rescue churn later.
          </p>
        </div>
        <div class="copy-card accent-orange">
          <p><strong>2. Satisfaction is the best leading signal.</strong></p>
          <p>
            Rating performance appears to capture more than simple timeliness. It likely reflects the overall impression
            of trust, fulfillment quality, and convenience.
          </p>
        </div>
        <div class="copy-card accent-blue">
          <p><strong>3. Promotions are weaker than service quality.</strong></p>
          <p>
            Discounts may help conversion or trial, but this dataset does not show that they create durable customer
            return behavior on their own.
          </p>
        </div>
      </div>
    </section>

    <section id="recommendations">
      <div class="section-head">
        <div>
          <h2>Recommendations</h2>
          <p>These recommendations prioritize what the analysis suggests will have the highest retention payoff.</p>
        </div>
      </div>
      <div class="grid-2">
        <div class="copy-card accent-teal">
          <h3 style="margin-bottom:12px;">Operational actions</h3>
          <ul class="recommendation-list">
            <li>Protect the first-order journey with tighter monitoring on delayed deliveries, especially orders at risk of slipping into the 15+ minute bucket.</li>
            <li>Use customer rating as an operational north-star metric for early retention, not just as a service score.</li>
            <li>Create a first-order recovery workflow for delayed or low-rated orders, such as service follow-up, apology credits, or support outreach.</li>
          </ul>
        </div>
        <div class="copy-card accent-violet">
          <h3 style="margin-bottom:12px;">Commercial actions</h3>
          <ul class="recommendation-list">
            <li>Reassess blanket promotion spend, since discounts do not appear to generate strong retention lift by themselves.</li>
            <li>Shift testing toward service-led retention strategies and measure whether operational improvements outperform discounting.</li>
            <li>Invest in better personalization data capture so future analysis can test CRM exposure, recommendations, and targeted offers directly.</li>
          </ul>
        </div>
      </div>
    </section>

    <section id="limits">
      <div class="section-head">
        <div>
          <h2>Limitations</h2>
          <p>The findings are useful and directionally strong, but they still sit inside a few important data constraints.</p>
        </div>
      </div>
      <div class="grid-3">
        <div class="copy-card accent-orange">
          <p><strong>No direct personalization data</strong></p>
          <p>Campaign exposure, recommendation impressions, push notifications, and loyalty interactions are not available, so the personalization half of Hypothesis 2 cannot be tested.</p>
        </div>
        <div class="copy-card accent-blue">
          <p><strong>Very high 90-day return rates</strong></p>
          <p>Because almost everyone in the eligible 90-day cohort returned, the adjusted logistic models at that window became unstable and should be interpreted carefully.</p>
        </div>
        <div class="copy-card accent-teal">
          <p><strong>Retention is inferred, not stored</strong></p>
          <p>The dataset does not contain a native churn label, so retention depends on how the return window is defined and on having enough time to observe it.</p>
        </div>
      </div>
    </section>

    <footer>
      Report built from the UrbanCart raw, interim, and processed datasets plus the approved hypothesis-testing outputs in this project workspace.
    </footer>
  </div>
</body>
</html>
"""
    return html


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(build_report(), encoding="utf-8")
    print(f"Wrote HTML report to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
