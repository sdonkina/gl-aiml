from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import mean, median


BASE_DIR = Path(__file__).resolve().parent.parent
INTERIM_DIR = BASE_DIR / "data" / "interim"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"
OUTPUT_PATH = REPORTS_DIR / "hypothesis_testing_results.txt"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def safe_float(value: str) -> float:
    return float(value) if value != "" else 0.0


def safe_int(value: str) -> int:
    return int(value) if value != "" else 0


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def two_sided_normal_pvalue(z_value: float) -> float:
    return max(0.0, min(1.0, 2.0 * (1.0 - normal_cdf(abs(z_value)))))


def gammaincc(a: float, x: float) -> float:
    if x < 0 or a <= 0:
        raise ValueError("Invalid arguments for gammaincc")
    if x == 0:
        return 1.0
    if x < a + 1.0:
        ap = a
        total = 1.0 / a
        delta = total
        for _ in range(200):
            ap += 1.0
            delta *= x / ap
            total += delta
            if abs(delta) < abs(total) * 1e-14:
                break
        gamser = total * math.exp(-x + a * math.log(x) - math.lgamma(a))
        return 1.0 - gamser
    b = x + 1.0 - a
    c = 1.0 / 1e-30
    d = 1.0 / b
    h = d
    for i in range(1, 200):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-30:
            d = 1e-30
        c = b + an / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def chi_square_test(table: list[list[int]]) -> tuple[float, int, float]:
    row_totals = [sum(row) for row in table]
    col_totals = [sum(table[row][col] for row in range(len(table))) for col in range(len(table[0]))]
    total = sum(row_totals)
    chi2 = 0.0
    for i, row_total in enumerate(row_totals):
        for j, col_total in enumerate(col_totals):
            expected = row_total * col_total / total
            if expected > 0:
                diff = table[i][j] - expected
                chi2 += (diff * diff) / expected
    degrees_freedom = (len(table) - 1) * (len(table[0]) - 1)
    p_value = gammaincc(degrees_freedom / 2.0, chi2 / 2.0)
    return chi2, degrees_freedom, p_value


def mann_whitney_u_test(group_a: list[float], group_b: list[float]) -> tuple[float, float, float]:
    combined = [(value, 0) for value in group_a] + [(value, 1) for value in group_b]
    combined.sort(key=lambda item: item[0])

    ranks = [0.0] * len(combined)
    tie_sizes: list[int] = []
    index = 0
    while index < len(combined):
        end = index + 1
        while end < len(combined) and combined[end][0] == combined[index][0]:
            end += 1
        avg_rank = (index + 1 + end) / 2.0
        for pos in range(index, end):
            ranks[pos] = avg_rank
        tie_sizes.append(end - index)
        index = end

    rank_sum_a = sum(ranks[pos] for pos, (_, group_id) in enumerate(combined) if group_id == 0)
    n_a = len(group_a)
    n_b = len(group_b)
    u_a = rank_sum_a - n_a * (n_a + 1) / 2.0
    u_b = n_a * n_b - u_a
    u_stat = min(u_a, u_b)

    mean_u = n_a * n_b / 2.0
    tie_correction = sum(tie ** 3 - tie for tie in tie_sizes)
    variance_u = (n_a * n_b / 12.0) * (
        (n_a + n_b + 1) - tie_correction / ((n_a + n_b) * (n_a + n_b - 1))
    )
    if variance_u <= 0:
        return u_stat, 0.0, 1.0
    continuity = 0.5 if u_stat > mean_u else -0.5
    z_value = (u_stat - mean_u + continuity) / math.sqrt(variance_u)
    p_value = two_sided_normal_pvalue(z_value)
    return u_stat, z_value, p_value


def invert_matrix(matrix: list[list[float]]) -> list[list[float]]:
    size = len(matrix)
    augmented = [row[:] + [1.0 if i == j else 0.0 for j in range(size)] for i, row in enumerate(matrix)]

    for col in range(size):
        pivot_row = max(range(col, size), key=lambda row_index: abs(augmented[row_index][col]))
        pivot_value = augmented[pivot_row][col]
        if abs(pivot_value) < 1e-12:
            raise ValueError("Matrix is singular and cannot be inverted")
        if pivot_row != col:
            augmented[col], augmented[pivot_row] = augmented[pivot_row], augmented[col]
        pivot_value = augmented[col][col]
        augmented[col] = [value / pivot_value for value in augmented[col]]
        for row_index in range(size):
            if row_index == col:
                continue
            factor = augmented[row_index][col]
            augmented[row_index] = [
                current - factor * pivot
                for current, pivot in zip(augmented[row_index], augmented[col])
            ]

    return [row[size:] for row in augmented]


def logistic_regression(
    rows: list[dict[str, object]],
    outcome_key: str,
    feature_builders: list[tuple[str, callable]],
    max_iter: int = 100,
) -> dict[str, object]:
    x_matrix: list[list[float]] = []
    y_vector: list[float] = []
    for row in rows:
        x_matrix.append([1.0] + [float(builder(row)) for _, builder in feature_builders])
        y_vector.append(float(row[outcome_key]))

    n_obs = len(x_matrix)
    n_params = len(x_matrix[0])
    beta = [0.0] * n_params
    converged = False

    for _ in range(max_iter):
        probabilities: list[float] = []
        weights: list[float] = []
        for x_row in x_matrix:
            linear = sum(coef * value for coef, value in zip(beta, x_row))
            linear = max(min(linear, 35.0), -35.0)
            probability = 1.0 / (1.0 + math.exp(-linear))
            probability = min(max(probability, 1e-9), 1.0 - 1e-9)
            probabilities.append(probability)
            weights.append(probability * (1.0 - probability))

        xtwx = [[0.0] * n_params for _ in range(n_params)]
        xty = [0.0] * n_params
        for x_row, y_val, probability, weight in zip(x_matrix, y_vector, probabilities, weights):
            for i in range(n_params):
                xty[i] += x_row[i] * (y_val - probability)
                for j in range(n_params):
                    xtwx[i][j] += x_row[i] * weight * x_row[j]

        try:
            xtwx_inv = invert_matrix(xtwx)
        except ValueError as exc:
            return {"error": str(exc), "n_obs": n_obs}

        step = [
            sum(xtwx_inv[i][j] * xty[j] for j in range(n_params))
            for i in range(n_params)
        ]
        beta = [coef + delta for coef, delta in zip(beta, step)]
        if max(abs(delta) for delta in step) < 1e-8:
            converged = True
            break

    if not converged:
        return {"error": "Logistic regression did not converge", "n_obs": n_obs}

    probabilities = []
    weights = []
    for x_row in x_matrix:
        linear = sum(coef * value for coef, value in zip(beta, x_row))
        linear = max(min(linear, 35.0), -35.0)
        probability = 1.0 / (1.0 + math.exp(-linear))
        probability = min(max(probability, 1e-9), 1.0 - 1e-9)
        probabilities.append(probability)
        weights.append(probability * (1.0 - probability))

    xtwx = [[0.0] * n_params for _ in range(n_params)]
    for x_row, weight in zip(x_matrix, weights):
        for i in range(n_params):
            for j in range(n_params):
                xtwx[i][j] += x_row[i] * weight * x_row[j]
    covariance = invert_matrix(xtwx)
    standard_errors = [math.sqrt(max(covariance[i][i], 0.0)) for i in range(n_params)]

    names = ["Intercept"] + [name for name, _ in feature_builders]
    coefficients = []
    for name, coef, se in zip(names, beta, standard_errors):
        z_value = coef / se if se > 0 else 0.0
        p_value = two_sided_normal_pvalue(z_value) if se > 0 else 1.0
        coefficients.append(
            {
                "term": name,
                "coef": coef,
                "std_err": se,
                "z": z_value,
                "p_value": p_value,
                "odds_ratio": math.exp(coef),
                "ci_low": math.exp(coef - 1.96 * se) if se > 0 else math.exp(coef),
                "ci_high": math.exp(coef + 1.96 * se) if se > 0 else math.exp(coef),
            }
        )

    log_likelihood = sum(
        y_val * math.log(probability) + (1.0 - y_val) * math.log(1.0 - probability)
        for y_val, probability in zip(y_vector, probabilities)
    )
    mean_outcome = sum(y_vector) / len(y_vector)
    null_log_likelihood = sum(
        y_val * math.log(mean_outcome) + (1.0 - y_val) * math.log(1.0 - mean_outcome)
        for y_val in y_vector
    )
    pseudo_r2 = 1.0 - (log_likelihood / null_log_likelihood)

    return {
        "n_obs": n_obs,
        "coefficients": coefficients,
        "log_likelihood": log_likelihood,
        "pseudo_r2": pseudo_r2,
    }


def benjamini_hochberg_adjust(pvalues: list[float]) -> list[float]:
    ranked = sorted(enumerate(pvalues), key=lambda item: item[1])
    adjusted = [0.0] * len(pvalues)
    running_min = 1.0
    total = len(pvalues)
    for rank, (original_index, pvalue) in enumerate(reversed(ranked), start=1):
        adjusted_value = min(1.0, pvalue * total / (total - rank + 1))
        running_min = min(running_min, adjusted_value)
        adjusted[original_index] = running_min
    return adjusted


def format_pvalue(p_value: float) -> str:
    if p_value < 0.0001:
        return "<0.0001"
    return f"{p_value:.4f}"


def format_rate(numerator: int, denominator: int) -> str:
    rate = numerator / denominator if denominator else 0.0
    return f"{numerator}/{denominator} ({rate:.1%})"


def load_first_order_analysis_rows() -> list[dict[str, object]]:
    order_rows = read_csv(INTERIM_DIR / "order_level_analysis_dataset.csv")
    first_orders = {row["customer_id"]: row for row in order_rows if row["customer_order_number"] == "1"}

    customer_rows = read_csv(PROCESSED_DIR / "customer_level_analysis_dataset.csv")
    merged_rows: list[dict[str, object]] = []
    for customer in customer_rows:
        customer_id = customer["customer_id"]
        if customer_id not in first_orders:
            continue
        order = first_orders[customer_id]
        merged_rows.append(
            {
                "customer_id": customer_id,
                "order_value_usd": safe_float(order["order_value_usd"]),
                "items_count": safe_int(order["items_count"]),
                "delay_minutes": safe_int(order["delay_minutes"]),
                "customer_rating": safe_int(order["customer_rating"]),
                "delivery_status": order["delivery_status"],
                "order_accuracy": order["order_accuracy"],
                "promo_used": order["promo_used"],
                "discount_amount_usd": safe_float(order["discount_amount_usd"]),
                "vendor_category": order["vendor_category"],
                "delivery_type": order["delivery_type"],
                "customer_segment": order["customer_segment"],
                "preferred_device": order["preferred_device"],
                "returned_after_first_order_30d": customer["returned_after_first_order_30d"],
                "returned_after_first_order_60d": customer["returned_after_first_order_60d"],
                "returned_after_first_order_90d": customer["returned_after_first_order_90d"],
                "full_30d_observation_flag": customer["full_30d_observation_flag"],
                "full_60d_observation_flag": customer["full_60d_observation_flag"],
                "full_90d_observation_flag": customer["full_90d_observation_flag"],
            }
        )
    return merged_rows


def eligible_rows(rows: list[dict[str, object]], window: int) -> list[dict[str, object]]:
    outcome_key = f"returned_after_first_order_{window}d"
    flag_key = f"full_{window}d_observation_flag"
    return [
        {
            **row,
            outcome_key: safe_int(str(row[outcome_key])),
        }
        for row in rows
        if str(row[flag_key]) == "1" and str(row[outcome_key]) != ""
    ]


def summarize_binary_groups(rows: list[dict[str, object]], group_key: str, outcome_key: str) -> list[str]:
    group_values = sorted({str(row[group_key]) for row in rows})
    lines = []
    for group_value in group_values:
        subset = [row for row in rows if str(row[group_key]) == group_value]
        retained = sum(int(row[outcome_key]) for row in subset)
        lines.append(f"- {group_key} = {group_value}: {format_rate(retained, len(subset))}")
    return lines


def summarize_numeric_by_outcome(rows: list[dict[str, object]], numeric_key: str, outcome_key: str) -> tuple[str, str]:
    retained = [float(row[numeric_key]) for row in rows if int(row[outcome_key]) == 1]
    not_retained = [float(row[numeric_key]) for row in rows if int(row[outcome_key]) == 0]
    retained_summary = f"retained mean={mean(retained):.2f}, median={median(retained):.2f}"
    not_summary = f"not retained mean={mean(not_retained):.2f}, median={median(not_retained):.2f}"
    return retained_summary, not_summary


def hypothesis_one_features() -> list[tuple[str, callable]]:
    return [
        ("delay_minutes", lambda row: row["delay_minutes"]),
        ("customer_rating", lambda row: row["customer_rating"]),
        ("accuracy_partial", lambda row: 1 if row["order_accuracy"] == "Partial" else 0),
        ("accuracy_wrong", lambda row: 1 if row["order_accuracy"] == "Wrong" else 0),
        ("order_value_usd", lambda row: row["order_value_usd"] / 10.0),
        ("items_count", lambda row: row["items_count"]),
        ("segment_regular", lambda row: 1 if row["customer_segment"] == "Regular" else 0),
        ("segment_premium", lambda row: 1 if row["customer_segment"] == "Premium" else 0),
        ("device_web", lambda row: 1 if row["preferred_device"] == "Web" else 0),
    ]


def hypothesis_two_features() -> list[tuple[str, callable]]:
    return [
        ("promo_used_yes", lambda row: 1 if row["promo_used"] == "Yes" else 0),
        ("order_value_usd", lambda row: row["order_value_usd"] / 10.0),
        ("items_count", lambda row: row["items_count"]),
        ("delay_minutes", lambda row: row["delay_minutes"]),
        ("customer_rating", lambda row: row["customer_rating"]),
        ("segment_regular", lambda row: 1 if row["customer_segment"] == "Regular" else 0),
        ("segment_premium", lambda row: 1 if row["customer_segment"] == "Premium" else 0),
        ("device_web", lambda row: 1 if row["preferred_device"] == "Web" else 0),
        ("delivery_type_scheduled", lambda row: 1 if row["delivery_type"] == "Scheduled" else 0),
        ("delivery_type_express", lambda row: 1 if row["delivery_type"] == "Express" else 0),
        ("vendor_grocery", lambda row: 1 if row["vendor_category"] == "Grocery" else 0),
        ("vendor_pharmacy", lambda row: 1 if row["vendor_category"] == "Pharmacy" else 0),
        ("vendor_retail", lambda row: 1 if row["vendor_category"] == "Retail" else 0),
    ]


def discount_depth_features() -> list[tuple[str, callable]]:
    return [
        ("discount_amount_usd", lambda row: row["discount_amount_usd"]),
        ("order_value_usd", lambda row: row["order_value_usd"] / 10.0),
        ("items_count", lambda row: row["items_count"]),
        ("delay_minutes", lambda row: row["delay_minutes"]),
        ("customer_rating", lambda row: row["customer_rating"]),
        ("segment_regular", lambda row: 1 if row["customer_segment"] == "Regular" else 0),
        ("segment_premium", lambda row: 1 if row["customer_segment"] == "Premium" else 0),
        ("device_web", lambda row: 1 if row["preferred_device"] == "Web" else 0),
    ]


def coefficient_line(model: dict[str, object], term_name: str) -> str:
    coefficients = model.get("coefficients", [])
    for coefficient in coefficients:
        if coefficient["term"] == term_name:
            return (
                f"- {term_name}: OR={coefficient['odds_ratio']:.3f}, "
                f"95% CI [{coefficient['ci_low']:.3f}, {coefficient['ci_high']:.3f}], "
                f"p={format_pvalue(coefficient['p_value'])}"
            )
    return f"- {term_name}: not available"


def build_report() -> str:
    rows = load_first_order_analysis_rows()
    windows = [30, 60, 90]
    lines = [
        "UrbanCart Hypothesis Testing Results",
        "",
        "Method note",
        "- Unit of analysis: customer first order only",
        "- Eligible cohorts exclude customers without a full observation window for the tested outcome",
        "- Primary outcome per section is first-order return within 30 days, with 60-day and 90-day robustness checks",
        "- Logistic regressions were estimated with a dependency-free Newton-Raphson implementation in Python",
        "",
    ]

    for window in windows:
        cohort = eligible_rows(rows, window)
        outcome_key = f"returned_after_first_order_{window}d"
        retained = sum(int(row[outcome_key]) for row in cohort)
        lines.extend(
            [
                f"Cohort overview: {window}-day window",
                f"- Eligible first-order customers: {len(cohort)}",
                f"- Retained within {window} days: {format_rate(retained, len(cohort))}",
                "",
            ]
        )

    for window in windows:
        cohort = eligible_rows(rows, window)
        outcome_key = f"returned_after_first_order_{window}d"
        lines.extend(
            [
                f"Hypothesis 1: Delivery reliability and {window}-day retention",
                "Descriptive view",
            ]
        )
        lines.extend(summarize_binary_groups(cohort, "delivery_status", outcome_key))
        lines.extend(summarize_binary_groups(cohort, "order_accuracy", outcome_key))
        delay_retained, delay_not = summarize_numeric_by_outcome(cohort, "delay_minutes", outcome_key)
        rating_retained, rating_not = summarize_numeric_by_outcome(cohort, "customer_rating", outcome_key)
        lines.append(f"- Delay minutes: {delay_retained}; {delay_not}")
        lines.append(f"- Customer rating: {rating_retained}; {rating_not}")

        status_table = [
            [
                sum(1 for row in cohort if row["delivery_status"] == "On-Time" and int(row[outcome_key]) == 1),
                sum(1 for row in cohort if row["delivery_status"] == "On-Time" and int(row[outcome_key]) == 0),
            ],
            [
                sum(1 for row in cohort if row["delivery_status"] == "Delayed" and int(row[outcome_key]) == 1),
                sum(1 for row in cohort if row["delivery_status"] == "Delayed" and int(row[outcome_key]) == 0),
            ],
        ]
        accuracy_categories = ["Correct", "Partial", "Wrong"]
        accuracy_table = [
            [
                sum(1 for row in cohort if row["order_accuracy"] == category and int(row[outcome_key]) == 1),
                sum(1 for row in cohort if row["order_accuracy"] == category and int(row[outcome_key]) == 0),
            ]
            for category in accuracy_categories
        ]
        chi2_status, df_status, p_status = chi_square_test(status_table)
        chi2_accuracy, df_accuracy, p_accuracy = chi_square_test(accuracy_table)
        delay_values_retained = [row["delay_minutes"] for row in cohort if int(row[outcome_key]) == 1]
        delay_values_not = [row["delay_minutes"] for row in cohort if int(row[outcome_key]) == 0]
        rating_values_retained = [row["customer_rating"] for row in cohort if int(row[outcome_key]) == 1]
        rating_values_not = [row["customer_rating"] for row in cohort if int(row[outcome_key]) == 0]
        u_delay, z_delay, p_delay = mann_whitney_u_test(delay_values_retained, delay_values_not)
        u_rating, z_rating, p_rating = mann_whitney_u_test(rating_values_retained, rating_values_not)
        adjusted = benjamini_hochberg_adjust([p_status, p_accuracy, p_delay, p_rating])
        lines.extend(
            [
                "Statistical tests",
                f"- Chi-square for delivery status vs retention: chi2={chi2_status:.3f}, df={df_status}, p={format_pvalue(p_status)}, BH-adjusted p={format_pvalue(adjusted[0])}",
                f"- Chi-square for order accuracy vs retention: chi2={chi2_accuracy:.3f}, df={df_accuracy}, p={format_pvalue(p_accuracy)}, BH-adjusted p={format_pvalue(adjusted[1])}",
                f"- Mann-Whitney U for delay minutes: U={u_delay:.1f}, z={z_delay:.3f}, p={format_pvalue(p_delay)}, BH-adjusted p={format_pvalue(adjusted[2])}",
                f"- Mann-Whitney U for customer rating: U={u_rating:.1f}, z={z_rating:.3f}, p={format_pvalue(p_rating)}, BH-adjusted p={format_pvalue(adjusted[3])}",
            ]
        )
        model = logistic_regression(cohort, outcome_key, hypothesis_one_features())
        lines.append("Adjusted logistic regression")
        if "error" in model:
            lines.append(f"- Model error: {model['error']}")
        else:
            lines.append(f"- n={model['n_obs']}, McFadden pseudo-R2={model['pseudo_r2']:.4f}")
            lines.append(coefficient_line(model, "delay_minutes"))
            lines.append(coefficient_line(model, "customer_rating"))
            lines.append(coefficient_line(model, "accuracy_partial"))
            lines.append(coefficient_line(model, "accuracy_wrong"))
        lines.append("")

    for window in windows:
        cohort = eligible_rows(rows, window)
        outcome_key = f"returned_after_first_order_{window}d"
        promo_users = [row for row in cohort if row["promo_used"] == "Yes"]
        lines.extend(
            [
                f"Hypothesis 2: Promotion-driven engagement and {window}-day retention",
                "Descriptive view",
            ]
        )
        lines.extend(summarize_binary_groups(cohort, "promo_used", outcome_key))
        discount_retained, discount_not = summarize_numeric_by_outcome(cohort, "discount_amount_usd", outcome_key)
        lines.append(f"- Discount amount across all first orders: {discount_retained}; {discount_not}")
        if promo_users:
            promo_discount_retained, promo_discount_not = summarize_numeric_by_outcome(promo_users, "discount_amount_usd", outcome_key)
            lines.append(
                f"- Discount amount among promo users only: {promo_discount_retained}; {promo_discount_not}"
            )

        promo_table = [
            [
                sum(1 for row in cohort if row["promo_used"] == "Yes" and int(row[outcome_key]) == 1),
                sum(1 for row in cohort if row["promo_used"] == "Yes" and int(row[outcome_key]) == 0),
            ],
            [
                sum(1 for row in cohort if row["promo_used"] == "No" and int(row[outcome_key]) == 1),
                sum(1 for row in cohort if row["promo_used"] == "No" and int(row[outcome_key]) == 0),
            ],
        ]
        chi2_promo, df_promo, p_promo = chi_square_test(promo_table)
        discount_values_retained = [row["discount_amount_usd"] for row in cohort if int(row[outcome_key]) == 1]
        discount_values_not = [row["discount_amount_usd"] for row in cohort if int(row[outcome_key]) == 0]
        u_discount_all, z_discount_all, p_discount_all = mann_whitney_u_test(discount_values_retained, discount_values_not)
        adjusted_tests = [p_promo, p_discount_all]
        if promo_users:
            promo_discount_retained_vals = [row["discount_amount_usd"] for row in promo_users if int(row[outcome_key]) == 1]
            promo_discount_not_vals = [row["discount_amount_usd"] for row in promo_users if int(row[outcome_key]) == 0]
            u_discount_promo, z_discount_promo, p_discount_promo = mann_whitney_u_test(
                promo_discount_retained_vals,
                promo_discount_not_vals,
            )
            adjusted_tests.append(p_discount_promo)
        else:
            u_discount_promo = z_discount_promo = p_discount_promo = None
        adjusted = benjamini_hochberg_adjust(adjusted_tests)
        lines.extend(
            [
                "Statistical tests",
                f"- Chi-square for promo usage vs retention: chi2={chi2_promo:.3f}, df={df_promo}, p={format_pvalue(p_promo)}, BH-adjusted p={format_pvalue(adjusted[0])}",
                f"- Mann-Whitney U for discount amount across all first orders: U={u_discount_all:.1f}, z={z_discount_all:.3f}, p={format_pvalue(p_discount_all)}, BH-adjusted p={format_pvalue(adjusted[1])}",
            ]
        )
        if u_discount_promo is not None:
            lines.append(
                f"- Mann-Whitney U for discount amount among promo users: U={u_discount_promo:.1f}, z={z_discount_promo:.3f}, p={format_pvalue(p_discount_promo)}, BH-adjusted p={format_pvalue(adjusted[2])}"
            )

        promo_model = logistic_regression(cohort, outcome_key, hypothesis_two_features())
        lines.append("Adjusted logistic regression")
        if "error" in promo_model:
            lines.append(f"- Model error: {promo_model['error']}")
        else:
            lines.append(f"- n={promo_model['n_obs']}, McFadden pseudo-R2={promo_model['pseudo_r2']:.4f}")
            lines.append(coefficient_line(promo_model, "promo_used_yes"))

        if promo_users:
            discount_model = logistic_regression(promo_users, outcome_key, discount_depth_features())
            lines.append("Promo-user discount-depth model")
            if "error" in discount_model:
                lines.append(f"- Model error: {discount_model['error']}")
            else:
                lines.append(f"- n={discount_model['n_obs']}, McFadden pseudo-R2={discount_model['pseudo_r2']:.4f}")
                lines.append(coefficient_line(discount_model, "discount_amount_usd"))
        lines.append("")

    lines.extend(
        [
            "Overall conclusion",
            "- Hypothesis 1 is supported in direction, with the strongest and most consistent signal coming from customer rating. Customers with delayed first orders also show lower retention descriptively, but delay minutes are not a stable adjusted predictor once the other covariates are included.",
            "- Hypothesis 2 is not supported for the promotion-driven portion of the question. Promo usage and discount depth do not show statistically reliable retention lifts in these first-order models.",
            "- Personalization still cannot be tested with the available data because there are no exposure fields for campaigns, recommendations, or CRM interactions.",
            "- The 90-day logistic models are less stable because nearly all eligible customers return within 90 days, leaving very few non-return cases for estimation.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    report = build_report()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(report, encoding="utf-8")
    print(f"Wrote hypothesis testing results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
