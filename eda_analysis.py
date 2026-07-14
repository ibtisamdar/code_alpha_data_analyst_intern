"""
TASK 2: EXPLORATORY DATA ANALYSIS (EDA)
-----------------------------------------
Loads any CSV and runs a structured EDA pass: structure/dtypes, missing
values, duplicates, descriptive stats, distributions, correlations, and
a couple of example hypothesis checks. Edit INPUT_CSV / COLUMN names for
your own dataset.

Run: python3 eda_analysis.py
"""
import pandas as pd
import numpy as np
from scipy import stats
import os

# ----------------------- CONFIG: edit for your dataset -----------------------
INPUT_CSV = "../sample_data/reviews_sample.csv"
NUMERIC_COL_OF_INTEREST = "rating"      # used in the hypothesis test example
GROUP_COL = "verified_purchase"         # used in the hypothesis test example
OUTPUT_DIR = "eda_output"
# -------------------------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main():
    df = pd.read_csv(INPUT_CSV)

    section("1. Shape & Structure")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print(df.dtypes)

    section("2. Missing Values")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_report = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
    print(missing_report[missing_report.missing_count > 0])

    section("3. Duplicate Rows")
    dupes = df.duplicated().sum()
    print(f"Duplicate rows: {dupes}")

    section("4. Descriptive Statistics (numeric columns)")
    print(df.describe())

    section("5. Descriptive Statistics (categorical columns)")
    cat_cols = df.select_dtypes(include=["object", "str"]).columns
    for col in cat_cols:
        print(f"\n-- {col} --")
        print(df[col].value_counts().head(10))

    section("6. Correlation Matrix (numeric columns)")
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.shape[1] >= 2:
        corr = numeric_df.corr()
        print(corr)
        corr.to_csv(f"{OUTPUT_DIR}/correlation_matrix.csv")
    else:
        print("Not enough numeric columns for a correlation matrix.")

    section("7. Outlier Check (IQR method, numeric columns)")
    for col in numeric_df.columns:
        q1, q3 = numeric_df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = numeric_df[(numeric_df[col] < lower) | (numeric_df[col] > upper)][col]
        print(f"{col}: {len(outliers)} potential outliers (bounds {lower:.2f} - {upper:.2f})")

    section("8. Example Hypothesis Test")
    if NUMERIC_COL_OF_INTEREST in df.columns and GROUP_COL in df.columns:
        groups = df[GROUP_COL].dropna().unique()
        if len(groups) == 2:
            g1 = df[df[GROUP_COL] == groups[0]][NUMERIC_COL_OF_INTEREST].dropna()
            g2 = df[df[GROUP_COL] == groups[1]][NUMERIC_COL_OF_INTEREST].dropna()
            t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)
            print(f"H0: mean({NUMERIC_COL_OF_INTEREST}) is equal across {GROUP_COL} groups {list(groups)}")
            print(f"t-statistic = {t_stat:.3f}, p-value = {p_val:.4f}")
            if p_val < 0.05:
                print("-> Statistically significant difference (reject H0 at alpha=0.05)")
            else:
                print("-> No statistically significant difference (fail to reject H0)")
        else:
            print(f"Skipped: '{GROUP_COL}' does not have exactly 2 groups.")
    else:
        print(f"Skipped: columns '{NUMERIC_COL_OF_INTEREST}' or '{GROUP_COL}' not found.")

    section("9. Data Issues Summary")
    issues = []
    if missing.sum() > 0:
        issues.append(f"{missing.sum()} missing values across {sum(missing > 0)} columns")
    if dupes > 0:
        issues.append(f"{dupes} duplicate rows")
    print("\n".join(issues) if issues else "No major issues detected.")

    # Save a clean summary report
    with open(f"{OUTPUT_DIR}/eda_summary.txt", "w") as f:
        f.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}\n")
        f.write(f"Duplicate rows: {dupes}\n")
        f.write(f"Missing values:\n{missing_report[missing_report.missing_count > 0]}\n")
    print(f"\nSummary saved to {OUTPUT_DIR}/eda_summary.txt")


if __name__ == "__main__":
    main()
