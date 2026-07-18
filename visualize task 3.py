"""
TASK 3: DATA VISUALIZATION
----------------------------
Loads a CSV and produces a set of portfolio-ready charts: distribution,
category comparison, trend over time, correlation heatmap, and a
combined dashboard image. Edit COLUMN names for your own dataset.

Run: python3 visualize.py
Output: PNG files in visuals_output/
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ----------------------- CONFIG: edit for your dataset -----------------------
INPUT_CSV = "../sample_data/reviews_sample.csv"
NUMERIC_COL = "rating"
CATEGORY_COL = "product_name"
DATE_COL = "review_date"
PRICE_COL = "price"
OUTPUT_DIR = "visuals_output"
# -------------------------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")


def main():
    df = pd.read_csv(INPUT_CSV, parse_dates=[DATE_COL])

    # 1. Distribution of the key numeric column
    plt.figure(figsize=(7, 5))
    sns.histplot(df[NUMERIC_COL], bins=5, kde=False, color="#4C72B0")
    plt.title(f"Distribution of {NUMERIC_COL}")
    plt.xlabel(NUMERIC_COL)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/01_distribution.png", dpi=150)
    plt.close()

    # 2. Average numeric value by category
    plt.figure(figsize=(8, 5))
    avg_by_cat = df.groupby(CATEGORY_COL)[NUMERIC_COL].mean().sort_values(ascending=False)
    sns.barplot(x=avg_by_cat.values, y=avg_by_cat.index, hue=avg_by_cat.index, palette="viridis", legend=False)
    plt.title(f"Average {NUMERIC_COL} by {CATEGORY_COL}")
    plt.xlabel(f"Average {NUMERIC_COL}")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/02_category_comparison.png", dpi=150)
    plt.close()

    # 3. Trend over time
    plt.figure(figsize=(9, 5))
    trend = df.set_index(DATE_COL).resample("ME")[NUMERIC_COL].mean()
    trend.plot(marker="o", color="#DD8452")
    plt.title(f"{NUMERIC_COL} Trend Over Time (Monthly Avg)")
    plt.xlabel("Date")
    plt.ylabel(f"Average {NUMERIC_COL}")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/03_trend_over_time.png", dpi=150)
    plt.close()

    # 4. Correlation heatmap
    numeric_df = df.select_dtypes(include="number")
    plt.figure(figsize=(6, 5))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/04_correlation_heatmap.png", dpi=150)
    plt.close()

    # 5. Price vs rating scatter (relationship view)
    if PRICE_COL in df.columns:
        plt.figure(figsize=(7, 5))
        sns.scatterplot(data=df, x=PRICE_COL, y=NUMERIC_COL, hue=CATEGORY_COL, alpha=0.7)
        plt.title(f"{PRICE_COL} vs {NUMERIC_COL}")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/05_price_vs_rating.png", dpi=150)
        plt.close()

    # 6. Combined dashboard (2x2 grid) for a single portfolio screenshot
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    sns.histplot(df[NUMERIC_COL], bins=5, ax=axes[0, 0], color="#4C72B0")
    axes[0, 0].set_title(f"Distribution of {NUMERIC_COL}")

    sns.barplot(x=avg_by_cat.values, y=avg_by_cat.index, hue=avg_by_cat.index, ax=axes[0, 1], palette="viridis", legend=False)
    axes[0, 1].set_title(f"Average {NUMERIC_COL} by {CATEGORY_COL}")

    trend.plot(marker="o", ax=axes[1, 0], color="#DD8452")
    axes[1, 0].set_title("Trend Over Time")

    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", center=0, ax=axes[1, 1])
    axes[1, 1].set_title("Correlation Heatmap")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/06_dashboard.png", dpi=150)
    plt.close()

    print(f"Saved 6 charts to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
