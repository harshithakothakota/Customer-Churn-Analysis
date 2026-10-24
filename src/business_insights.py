
from pathlib import Path

import pandas as pd


def generate_business_insights(df, segment_results, output_dir):
    """Generate evidence-based churn findings and recommendations."""

    output_dir = Path(output_dir)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    overall_churn = df["ChurnFlag"].mean() * 100

    churned = df[df["ChurnFlag"] == 1]
    retained = df[df["ChurnFlag"] == 0]

    churned_avg_tenure = churned["tenure"].mean()
    retained_avg_tenure = retained["tenure"].mean()

    churned_avg_charges = churned["MonthlyCharges"].mean()
    retained_avg_charges = retained["MonthlyCharges"].mean()

    tenure_gap = retained_avg_tenure - churned_avg_tenure
    charge_gap = churned_avg_charges - retained_avg_charges

    report_lines = [
        "CUSTOMER CHURN ANALYSIS: BUSINESS INSIGHTS",
        "=" * 48,
        "",
        "DATASET OVERVIEW",
        f"Customers analyzed: {len(df):,}",
        f"Customers churned: {len(churned):,}",
        f"Customers retained: {len(retained):,}",
        f"Overall churn rate: {overall_churn:.2f}%",
        "",
        "KEY FINDINGS",
        (
            f"1. Churned customers averaged "
            f"{churned_avg_tenure:.2f} months of tenure, "
            f"compared with {retained_avg_tenure:.2f} "
            f"months for retained customers."
        ),
        (
            f"2. Churned customers paid an average of "
            f"${churned_avg_charges:.2f} per month, "
            f"compared with ${retained_avg_charges:.2f} "
            f"for retained customers."
        ),
        (
            f"3. Retained customers had "
            f"{tenure_gap:.2f} more months of average tenure."
        ),
        (
            f"4. Churned customers paid "
            f"${charge_gap:.2f} more per month on average."
        ),
        "",
        "RECOMMENDATIONS",
        (
            "1. Investigate early-tenure churn and test "
            "onboarding or customer-support improvements."
        ),
        (
            "2. Review pricing, service usage, and perceived "
            "value among higher-charge customers."
        ),
        (
            "3. Investigate high-churn contract and payment "
            "segments before designing targeted interventions."
        ),
        (
            "4. Measure retention campaign outcomes against "
            "a suitable comparison group before scaling."
        ),
        "",
        "HIGHEST-CHURN CUSTOMER SEGMENTS",
    ]

    # Add the highest-churn segments when the required columns exist.
    required_segment_columns = {
        "segment_dimension",
        "customer_count",
        "churn_rate_percent",
    }

    if (
        isinstance(segment_results, pd.DataFrame)
        and not segment_results.empty
        and required_segment_columns.issubset(segment_results.columns)
    ):
        top_segments = (
            segment_results
            .sort_values("churn_rate_percent", ascending=False)
            .head(5)
        )

        for _, row in top_segments.iterrows():
            dimension = row["segment_dimension"]

            # The segment value column may have different names.
            if "segment_value" in segment_results.columns:
                segment_value = row["segment_value"]
            elif dimension in segment_results.columns:
                segment_value = row[dimension]
            elif "segment" in segment_results.columns:
                segment_value = row["segment"]
            else:
                segment_value = "Value unavailable"

            report_lines.append(
                f"- {dimension}: {segment_value} | "
                f"Customers: {int(row['customer_count']):,} | "
                f"Churn rate: {row['churn_rate_percent']:.2f}%"
            )
    else:
        report_lines.append(
            "Segment details could not be generated from the "
            "provided segment results."
        )

    report_lines.extend([
        "",
        "LIMITATIONS",
        (
            "These are observational findings and do not establish "
            "causation. Churn rates describe historical customers; "
            "they are not predictions of future churn."
        ),
        (
            "Records with missing essential numeric values were "
            "excluded during cleaning."
        ),
    ])

    report_path = reports_dir / "business_insights.txt"
    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("\nBusiness insights saved to:", report_path)

    return report_path
