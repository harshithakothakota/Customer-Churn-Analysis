
from pathlib import Path

import pandas as pd


def analyze_customer_segments(df, output_dir):
    """
    Compare churn rates across customer segments and
    identify high-churn groups for further investigation.
    """
    output_dir = Path(output_dir)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    segment_columns = [
        "Contract",
        "InternetService",
        "PaymentMethod",
        "TechSupport",
        "OnlineSecurity",
        "PaperlessBilling",
        "SeniorCitizen",
    ]

    segment_results = []

    for column in segment_columns:
        grouped = (
            df.groupby(column, dropna=False, observed=True)
            .agg(
                customer_count=("customerID", "count"),
                churned_customers=("ChurnFlag", "sum"),
                churn_rate=("ChurnFlag", "mean"),
                average_monthly_charges=("MonthlyCharges", "mean"),
                average_tenure=("tenure", "mean"),
            )
            .reset_index()
        )

        # Give every category value the same column name.
        grouped = grouped.rename(columns={column: "segment_value"})
        grouped.insert(0, "segment_dimension", column)

        grouped["churn_rate_percent"] = (
            grouped["churn_rate"] * 100
        ).round(2)

        grouped["average_monthly_charges"] = (
            grouped["average_monthly_charges"].round(2)
        )

        grouped["average_tenure"] = (
            grouped["average_tenure"].round(2)
        )

        segment_results.append(
            grouped[
                [
                    "segment_dimension",
                    "segment_value",
                    "customer_count",
                    "churned_customers",
                    "churn_rate",
                    "churn_rate_percent",
                    "average_monthly_charges",
                    "average_tenure",
                ]
            ]
        )

    all_segments = pd.concat(
        segment_results,
        ignore_index=True,
    )

    all_segments.to_csv(
        reports_dir / "customer_segment_analysis.csv",
        index=False,
    )

    # Keep segments with at least 50 customers for more
    # meaningful comparisons.
    priority_segments = all_segments[
        all_segments["customer_count"] >= 50
    ].sort_values(
        ["churn_rate", "customer_count"],
        ascending=[False, False],
    ).copy()

    priority_segments.to_csv(
        reports_dir / "high_churn_segments.csv",
        index=False,
    )

    print("\n--- HIGHEST-CHURN CUSTOMER SEGMENTS ---")

    display_columns = [
        "segment_dimension",
        "segment_value",
        "customer_count",
        "churn_rate_percent",
    ]

    print(
        priority_segments[display_columns]
        .head(10)
        .to_string(index=False)
    )

    print("\nSegment reports saved to:", reports_dir)

    return all_segments, priority_segments
