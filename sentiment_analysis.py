"""
TASK 4: SENTIMENT ANALYSIS
-----------------------------
Lexicon-based sentiment classification (VADER) on a text column.
Classifies each row as positive / negative / neutral, then summarizes
and visualizes the results. Edit TEXT_COL for your own dataset.

Run: python3 sentiment_analysis.py
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ----------------------- CONFIG: edit for your dataset -----------------------
INPUT_CSV = "../sample_data/reviews_sample.csv"
TEXT_COL = "review_text"
CATEGORY_COL = "product_name"   # optional, set to None to skip group breakdown
OUTPUT_DIR = "sentiment_output"
# -------------------------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
analyzer = SentimentIntensityAnalyzer()


def classify(text):
    score = analyzer.polarity_scores(str(text))["compound"]
    if score >= 0.05:
        return "positive", score
    elif score <= -0.05:
        return "negative", score
    return "neutral", score


def main():
    df = pd.read_csv(INPUT_CSV)

    results = df[TEXT_COL].apply(classify)
    df["sentiment"] = results.apply(lambda x: x[0])
    df["sentiment_score"] = results.apply(lambda x: x[1])

    print("Sentiment distribution:")
    print(df["sentiment"].value_counts())
    print(f"\nAverage sentiment score: {df['sentiment_score'].mean():.3f}")

    df.to_csv(f"{OUTPUT_DIR}/sentiment_results.csv", index=False)

    # Chart 1: overall sentiment distribution
    plt.figure(figsize=(6, 5))
    order = ["positive", "neutral", "negative"]
    colors = {"positive": "#55A868", "neutral": "#8C8C8C", "negative": "#C44E52"}
    counts = df["sentiment"].value_counts().reindex(order)
    plt.bar(counts.index, counts.values, color=[colors[s] for s in counts.index])
    plt.title("Overall Sentiment Distribution")
    plt.ylabel("Number of Reviews")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/01_sentiment_distribution.png", dpi=150)
    plt.close()

    # Chart 2: sentiment by category, if provided
    if CATEGORY_COL and CATEGORY_COL in df.columns:
        plt.figure(figsize=(9, 5))
        cross = pd.crosstab(df[CATEGORY_COL], df["sentiment"], normalize="index") * 100
        cross = cross.reindex(columns=order, fill_value=0)
        cross.plot(kind="bar", stacked=True, color=[colors[s] for s in order])
        plt.title(f"Sentiment Breakdown by {CATEGORY_COL}")
        plt.ylabel("% of Reviews")
        plt.legend(title="Sentiment")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/02_sentiment_by_category.png", dpi=150)
        plt.close()

    # Show a few example classified reviews
    print("\nSample classified reviews:")
    print(df[[TEXT_COL, "sentiment", "sentiment_score"]].sample(5, random_state=1).to_string(index=False))

    print(f"\nSaved results and charts to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
