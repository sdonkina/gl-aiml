from __future__ import annotations

import csv
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from statistics import mean


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
INTERIM_DIR = BASE_DIR / "data" / "interim"
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def safe_mean(values: list[float]) -> float:
    return round(mean(values), 2) if values else 0.0


def safe_ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def build_order_level_dataset(
    customers: list[dict[str, str]],
    orders: list[dict[str, str]],
    deliveries: list[dict[str, str]],
) -> list[dict[str, object]]:
    customers_by_id = {row["customer_id"]: row for row in customers}
    deliveries_by_order_id = {row["order_id"]: row for row in deliveries}

    orders_by_customer: dict[str, list[dict[str, str]]] = defaultdict(list)
    for order in orders:
        orders_by_customer[order["customer_id"]].append(order)

    for customer_orders in orders_by_customer.values():
        customer_orders.sort(key=lambda row: parse_date(row["order_date"]))

    enriched_orders: list[dict[str, object]] = []

    for customer_id, customer_orders in orders_by_customer.items():
        customer = customers_by_id[customer_id]
        for index, order in enumerate(customer_orders):
            delivery = deliveries_by_order_id[order["order_id"]]
            order_date = parse_date(order["order_date"])
            signup_date = parse_date(customer["signup_date"])
            next_order_date = ""
            days_to_next_order = ""
            returned_within_30d = 0
            returned_within_60d = 0
            returned_within_90d = 0

            if index + 1 < len(customer_orders):
                next_order = customer_orders[index + 1]
                next_order_date = next_order["order_date"]
                days_gap = (parse_date(next_order["order_date"]) - order_date).days
                days_to_next_order = days_gap
                returned_within_30d = int(days_gap <= 30)
                returned_within_60d = int(days_gap <= 60)
                returned_within_90d = int(days_gap <= 90)

            enriched_orders.append(
                {
                    "order_id": order["order_id"],
                    "customer_id": customer_id,
                    "order_date": order["order_date"],
                    "signup_date": customer["signup_date"],
                    "customer_tenure_at_order_days": (order_date - signup_date).days,
                    "city": customer["city"],
                    "state": customer["state"],
                    "age_group": customer["age_group"],
                    "gender": customer["gender"],
                    "customer_segment": customer["customer_segment"],
                    "preferred_device": customer["preferred_device"],
                    "vendor_id": order["vendor_id"],
                    "vendor_category": order["vendor_category"],
                    "delivery_type": order["delivery_type"],
                    "items_count": int(order["items_count"]),
                    "order_value_usd": float(order["order_value_usd"]),
                    "promo_used": order["promo_used"],
                    "discount_amount_usd": float(order["discount_amount_usd"]),
                    "delivery_id": delivery["delivery_id"],
                    "promised_time_min": int(delivery["promised_time_min"]),
                    "actual_time_min": int(delivery["actual_time_min"]),
                    "delay_minutes": int(delivery["delay_minutes"]),
                    "delivery_status": delivery["delivery_status"],
                    "order_accuracy": delivery["order_accuracy"],
                    "customer_rating": int(delivery["customer_rating"]),
                    "customer_order_number": index + 1,
                    "next_order_date": next_order_date,
                    "days_to_next_order": days_to_next_order,
                    "returned_within_30d": returned_within_30d,
                    "returned_within_60d": returned_within_60d,
                    "returned_within_90d": returned_within_90d,
                }
            )

    enriched_orders.sort(key=lambda row: (row["customer_id"], row["order_date"], row["order_id"]))
    return enriched_orders


def build_customer_level_dataset(
    customers: list[dict[str, str]],
    enriched_orders: list[dict[str, object]],
) -> list[dict[str, object]]:
    orders_by_customer: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in enriched_orders:
        orders_by_customer[str(row["customer_id"])].append(row)

    dataset_end_date = max(parse_date(str(row["order_date"])) for row in enriched_orders)
    customer_rows: list[dict[str, object]] = []

    for customer in customers:
        customer_id = customer["customer_id"]
        customer_orders = sorted(
            orders_by_customer.get(customer_id, []),
            key=lambda row: parse_date(str(row["order_date"])),
        )

        signup_date = parse_date(customer["signup_date"])
        total_orders = len(customer_orders)

        if total_orders == 0:
            customer_rows.append(
                {
                    "customer_id": customer_id,
                    "signup_date": customer["signup_date"],
                    "city": customer["city"],
                    "state": customer["state"],
                    "age_group": customer["age_group"],
                    "gender": customer["gender"],
                    "customer_segment": customer["customer_segment"],
                    "preferred_device": customer["preferred_device"],
                    "tenure_days_at_dataset_end": (dataset_end_date - signup_date).days,
                    "first_order_date": "",
                    "last_order_date": "",
                    "days_since_last_order": "",
                    "active_span_days": 0,
                    "total_orders": 0,
                    "repeat_customer_flag": 0,
                    "total_order_value_usd": 0.0,
                    "avg_order_value_usd": 0.0,
                    "avg_items_per_order": 0.0,
                    "promo_orders_count": 0,
                    "promo_order_rate": 0.0,
                    "total_discount_amount_usd": 0.0,
                    "avg_discount_amount_usd": 0.0,
                    "unique_vendors": 0,
                    "unique_vendor_categories": 0,
                    "unique_delivery_types": 0,
                    "multi_category_user_flag": 0,
                    "multi_delivery_type_user_flag": 0,
                    "restaurant_orders": 0,
                    "grocery_orders": 0,
                    "pharmacy_orders": 0,
                    "retail_orders": 0,
                    "on_demand_orders": 0,
                    "scheduled_orders": 0,
                    "express_orders": 0,
                    "on_time_deliveries": 0,
                    "delayed_deliveries": 0,
                    "on_time_delivery_rate": 0.0,
                    "avg_delay_minutes": 0.0,
                    "max_delay_minutes": 0,
                    "correct_orders": 0,
                    "partial_orders": 0,
                    "wrong_orders": 0,
                    "correct_order_rate": 0.0,
                    "avg_customer_rating": 0.0,
                    "returned_after_first_order_30d": 0,
                    "returned_after_first_order_60d": 0,
                    "returned_after_first_order_90d": 0,
                    "eligible_for_30d_window": 0,
                    "eligible_for_60d_window": 0,
                    "eligible_for_90d_window": 0,
                }
            )
            continue

        first_order = customer_orders[0]
        last_order = customer_orders[-1]
        first_order_date = parse_date(str(first_order["order_date"]))
        last_order_date = parse_date(str(last_order["order_date"]))

        order_values = [float(row["order_value_usd"]) for row in customer_orders]
        items_counts = [int(row["items_count"]) for row in customer_orders]
        discount_amounts = [float(row["discount_amount_usd"]) for row in customer_orders]
        promo_orders_count = sum(1 for row in customer_orders if row["promo_used"] == "Yes")
        vendor_categories = Counter(str(row["vendor_category"]) for row in customer_orders)
        delivery_types = Counter(str(row["delivery_type"]) for row in customer_orders)
        delivery_statuses = Counter(str(row["delivery_status"]) for row in customer_orders)
        order_accuracy = Counter(str(row["order_accuracy"]) for row in customer_orders)
        ratings = [int(row["customer_rating"]) for row in customer_orders]
        delays = [int(row["delay_minutes"]) for row in customer_orders]
        unique_vendors = len({str(row["vendor_id"]) for row in customer_orders})

        full_30d_observation_flag = int((dataset_end_date - first_order_date).days >= 30)
        full_60d_observation_flag = int((dataset_end_date - first_order_date).days >= 60)
        full_90d_observation_flag = int((dataset_end_date - first_order_date).days >= 90)

        returned_after_first_order_30d = int(first_order["returned_within_30d"])
        returned_after_first_order_60d = int(first_order["returned_within_60d"])
        returned_after_first_order_90d = int(first_order["returned_within_90d"])

        # Leave outcomes blank when the customer has not returned and the full
        # observation window has not elapsed yet. This avoids treating censored
        # customers as confirmed non-returners.
        if not returned_after_first_order_30d and not full_30d_observation_flag:
            returned_after_first_order_30d = ""
        if not returned_after_first_order_60d and not full_60d_observation_flag:
            returned_after_first_order_60d = ""
        if not returned_after_first_order_90d and not full_90d_observation_flag:
            returned_after_first_order_90d = ""

        customer_rows.append(
            {
                "customer_id": customer_id,
                "signup_date": customer["signup_date"],
                "city": customer["city"],
                "state": customer["state"],
                "age_group": customer["age_group"],
                "gender": customer["gender"],
                "customer_segment": customer["customer_segment"],
                "preferred_device": customer["preferred_device"],
                "tenure_days_at_dataset_end": (dataset_end_date - signup_date).days,
                "first_order_date": first_order["order_date"],
                "last_order_date": last_order["order_date"],
                "days_since_last_order": (dataset_end_date - last_order_date).days,
                "active_span_days": (last_order_date - first_order_date).days,
                "total_orders": total_orders,
                "repeat_customer_flag": int(total_orders > 1),
                "total_order_value_usd": round(sum(order_values), 2),
                "avg_order_value_usd": safe_mean(order_values),
                "avg_items_per_order": safe_mean(items_counts),
                "promo_orders_count": promo_orders_count,
                "promo_order_rate": safe_ratio(promo_orders_count, total_orders),
                "total_discount_amount_usd": round(sum(discount_amounts), 2),
                "avg_discount_amount_usd": safe_mean(discount_amounts),
                "unique_vendors": unique_vendors,
                "unique_vendor_categories": len(vendor_categories),
                "unique_delivery_types": len(delivery_types),
                "multi_category_user_flag": int(len(vendor_categories) > 1),
                "multi_delivery_type_user_flag": int(len(delivery_types) > 1),
                "restaurant_orders": vendor_categories.get("Restaurant", 0),
                "grocery_orders": vendor_categories.get("Grocery", 0),
                "pharmacy_orders": vendor_categories.get("Pharmacy", 0),
                "retail_orders": vendor_categories.get("Retail", 0),
                "on_demand_orders": delivery_types.get("On-Demand", 0),
                "scheduled_orders": delivery_types.get("Scheduled", 0),
                "express_orders": delivery_types.get("Express", 0),
                "on_time_deliveries": delivery_statuses.get("On-Time", 0),
                "delayed_deliveries": delivery_statuses.get("Delayed", 0),
                "on_time_delivery_rate": safe_ratio(delivery_statuses.get("On-Time", 0), total_orders),
                "avg_delay_minutes": safe_mean(delays),
                "max_delay_minutes": max(delays),
                "correct_orders": order_accuracy.get("Correct", 0),
                "partial_orders": order_accuracy.get("Partial", 0),
                "wrong_orders": order_accuracy.get("Wrong", 0),
                "correct_order_rate": safe_ratio(order_accuracy.get("Correct", 0), total_orders),
                "avg_customer_rating": safe_mean(ratings),
                "returned_after_first_order_30d": returned_after_first_order_30d,
                "returned_after_first_order_60d": returned_after_first_order_60d,
                "returned_after_first_order_90d": returned_after_first_order_90d,
                "full_30d_observation_flag": full_30d_observation_flag,
                "full_60d_observation_flag": full_60d_observation_flag,
                "full_90d_observation_flag": full_90d_observation_flag,
            }
        )

    customer_rows.sort(key=lambda row: row["customer_id"])
    return customer_rows


def main() -> None:
    customers = read_csv(RAW_DIR / "customers.csv")
    orders = read_csv(RAW_DIR / "orders.csv")
    deliveries = read_csv(RAW_DIR / "deliveries.csv")

    enriched_orders = build_order_level_dataset(customers, orders, deliveries)
    customer_features = build_customer_level_dataset(customers, enriched_orders)

    order_level_fieldnames = list(enriched_orders[0].keys())
    customer_level_fieldnames = list(customer_features[0].keys())

    write_csv(INTERIM_DIR / "order_level_analysis_dataset.csv", enriched_orders, order_level_fieldnames)
    write_csv(PROCESSED_DIR / "customer_level_analysis_dataset.csv", customer_features, customer_level_fieldnames)

    print(f"Created {len(enriched_orders)} order-level rows")
    print(f"Created {len(customer_features)} customer-level rows")
    print(f"Dataset end date: {max(row['order_date'] for row in enriched_orders)}")


if __name__ == "__main__":
    main()
