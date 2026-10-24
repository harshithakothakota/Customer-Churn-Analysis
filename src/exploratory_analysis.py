
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid", palette="Set2")


def analyze_churn(df, output_dir):
    """
    Perform exploratory analysis and save charts and summary tables.
    """
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    reports_dir = output_dir / "reports"

    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Overall churn distribution
    churn_counts = df["Churn"].value_counts().reindex(
        ["No", "Yes"], fill_value=0
    )
    total_customers = churn_counts.sum()

    churn_percentages = (
        churn_counts / total_customers * 100
    )

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        x=churn_counts.index,
        y=churn_counts.values,
    )
    ax.set_title("Customer Churn Distribution")
    ax.set_xlabel("Churn Status")
    ax.set_ylabel("Number of Customers")

    for bar, count, percentage in zip(
        ax.patches,
        churn_counts.values,
        churn_percentages.values,
    ):
        ax.annotate(
            f"{count:,}\n({percentage:.1f}%)",
            (
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
            ),
            ha="center",
            va="bottom",
            xytext=(0, 5),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_ylim(0, churn_counts.max() * 1.12)

    plt.tight_layout()
    plt.savefig(figures_dir / "churn_distribution.png", dpi=300)
    plt.close()

    # 2. Churn rate by contract type
    contract_churn = (
        df.groupby("Contract", observed=True)["ChurnFlag"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
        .rename("ChurnRate")
        .reset_index()
    )

    contract_churn.to_csv(
        reports_dir / "churn_by_contract.csv", index=False
    )

    plt.figure(figsize=(9, 5))
    ax = sns.barplot(
        data=contract_churn,
        x="Contract",
        y="ChurnRate",
    )
    ax.set_title("Churn Rate by Contract Type")
    ax.set_xlabel("Contract Type")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_ylim(0, contract_churn["ChurnRate"].max() * 1.12)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%")

    plt.tight_layout()
    plt.savefig(figures_dir / "churn_by_contract.png", dpi=300)
    plt.close()

    # 3. Churn rate by tenure group
    tenure_churn = (
        df.groupby("TenureGroup", observed=False)["ChurnFlag"]
        .mean()
        .mul(100)
        .rename("ChurnRate")
        .reset_index()
    )

    tenure_churn.to_csv(
        reports_dir / "churn_by_tenure.csv", index=False
    )

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(
        data=tenure_churn,
        x="TenureGroup",
        y="ChurnRate",
    )
    ax.set_title("Churn Rate Across Customer Tenure Groups")
    ax.set_xlabel("Tenure")
    ax.set_ylabel("Churn Rate (%)")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", padding=4)

    ax.set_ylim(0, tenure_churn["ChurnRate"].max() * 1.12)
    plt.xticks(rotation=20)

    plt.tight_layout()
    plt.savefig(figures_dir / "churn_by_tenure.png", dpi=300)
    plt.close()

    # 4. Monthly charges comparison
    plt.figure(figsize=(9, 5))
    ax = sns.boxplot(
        data=df,
        x="Churn",
        y="MonthlyCharges",
        hue="Churn",
        legend=False,
    )

    medians = df.groupby("Churn")["MonthlyCharges"].median()

    for position, churn_status in enumerate(["No", "Yes"]):
        median_value = medians[churn_status]
        ax.annotate(
            f"Median: {median_value:.2f}",
            (position, median_value),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )
    plt.title("Monthly Charges: Churned vs Retained Customers")
    plt.xlabel("Churn Status")
    plt.ylabel("Monthly Charges")
    plt.tight_layout()
    plt.savefig(figures_dir / "monthly_charges_by_churn.png", dpi=300)
    plt.close()

    # 5. Correlation heatmap for numerical variables
    numeric_df = df[
        ["tenure", "MonthlyCharges", "TotalCharges", "ChurnFlag"]
    ]

    correlation = numeric_df.corr()

    correlation.to_csv(reports_dir / "correlation_matrix.csv")

    mask = pd.DataFrame(
        False,
        index=correlation.index,
        columns=correlation.columns,
    )

    for i in range(len(correlation)):
        mask.iloc[i, i] = True

        plt.figure(figsize=(9, 7))

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )

    plt.title("Correlation Heatmap", pad=15)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(figures_dir / "correlation_heatmap.png", dpi=300)
    plt.close()

    # 6. Summary metrics
    churned = df[df["ChurnFlag"] == 1]
    retained = df[df["ChurnFlag"] == 0]

    summary = {
        "total_customers": len(df),
        "churned_customers": len(churned),
        "retained_customers": len(retained),
        "overall_churn_rate_percent": round(
            df["ChurnFlag"].mean() * 100, 2
        ),
        "average_tenure_churned": round(
            churned["tenure"].mean(), 2
        ),
        "average_tenure_retained": round(
            retained["tenure"].mean(), 2
        ),
        "average_monthly_charges_churned": round(
            churned["MonthlyCharges"].mean(), 2
        ),
        "average_monthly_charges_retained": round(
            retained["MonthlyCharges"].mean(), 2
        ),
    }

    pd.DataFrame(
        summary.items(), columns=["Metric", "Value"]
    ).to_csv(reports_dir / "eda_summary.csv", index=False)

    print("\n--- EDA SUMMARY ---")
    for metric, value in summary.items():
        print(f"{metric}: {value}")

    print(f"\nCharts saved to: {figures_dir}")
    print(f"Reports saved to: {reports_dir}")

    return summary
