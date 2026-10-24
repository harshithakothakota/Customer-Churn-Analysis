from src.customer_segmentation import analyze_customer_segments
from pathlib import Path
from src.business_insights import generate_business_insights
from src.data_cleaning import (
    load_and_clean_data,
    generate_quality_report,
)
from src.exploratory_analysis import analyze_churn


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "customer_churn.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

# Step 1: Load and clean the dataset.
df = load_and_clean_data(DATA_PATH)

print("\n--- CLEANED DATASET ---")
print("Shape:", df.shape)

# Step 2: Print data-quality metrics.
print("\n--- DATA QUALITY REPORT ---")
for metric, value in generate_quality_report(df).items():
    print(f"{metric}: {value}")


# Step 3: Perform exploratory data analysis.
analyze_churn(df, OUTPUT_DIR)

# Step 4: Analyze customer segments.
segment_results, priority_segments = analyze_customer_segments(
    df, OUTPUT_DIR
)

# Step 5: Generate business insights.
generate_business_insights(
    df, segment_results, OUTPUT_DIR
)

print("\nProject analysis completed successfully!")
