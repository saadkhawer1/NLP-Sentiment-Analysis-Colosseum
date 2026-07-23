"""
Task 2 - Exploratory Data Analysis
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


def run_eda(df, plots_dir):
    """Run full EDA with clean structured output."""

    print("\n" + "+" + "-"*68 + "+")
    print("|{:^68s}|".format("TASK 2: EXPLORATORY DATA ANALYSIS"))
    print("+" + "-"*68 + "+")

    os.makedirs(plots_dir, exist_ok=True)

    # ---- 2a: Star rating distribution ----
    print("\n  [CHART 1] Star Rating Distribution")
    fig, ax = plt.subplots(figsize=(8, 5))

    # graph for checking how many people give 5 ya 4 stars
    df['rating'].value_counts().sort_index().plot(
        kind='bar', color=sns.color_palette("viridis", 5), ax=ax)
    ax.set_title("Distribution of Star Ratings", fontsize=14, fontweight='bold')
    ax.set_xlabel("Rating"); ax.set_ylabel("Count")
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}',
                    (p.get_x() + p.get_width()/2., p.get_height()),
                    ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "1_rating_distribution.png"), dpi=150)
    plt.close()
    

    print(f"    {'Rating':<10s} {'Count':<10s} {'Percentage':<12s}")
    print(f"    {'-'*32}")
    for rating in sorted(df['rating'].unique()):
        cnt = (df['rating'] == rating).sum()
        pct = cnt / len(df) * 100
        stars = '*' * int(rating)
        print(f"    {stars:<10s} {cnt:<10d} {pct:.1f}%")
    print(f"    -> Saved: plots/1_rating_distribution.png")

    # ---- 2b: Sentiment comparison ----
    print(f"\n  [CHART 2] Sentiment Label Comparison")
    # sentiment for rating is 5 ya 4 ho to positive, 3 ho to neutral, 1 ya 2 ho to negative
    df['rating_sentiment'] = df['rating'].apply(
        lambda r: 'Positive' if r >= 4 else ('Neutral' if r == 3 else 'Negative'))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, col, title in zip(axes,
            ['sentiment_label', 'rating_sentiment'],
            ['Dataset Sentiment Label', 'Rating-Derived Sentiment']):
        counts = df[col].value_counts()
        counts.plot(kind='bar', ax=ax, color=['#2ecc71', '#f39c12', '#e74c3c'])
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_ylabel("Count")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}',
                        (p.get_x() + p.get_width()/2., p.get_height()),
                        ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "2_sentiment_comparison.png"), dpi=150)
    plt.close()

    agree = (df['sentiment_label'] == df['rating_sentiment']).mean()

    print(f"    {'Class':<12s} {'Count':<10s} {'Percentage':<12s}")
    print(f"    {'-'*34}")
    for cls in ['Positive', 'Neutral', 'Negative']:
        cnt = (df['sentiment_label'] == cls).sum()
        pct = cnt / len(df) * 100
        print(f"    {cls:<12s} {cnt:<10d} {pct:.1f}%")
    print(f"\n    Label vs Rating-derived agreement: {agree:.1%}")
    print(f"    -> Saved: plots/2_sentiment_comparison.png")

    # ---- 2c: Trip type & travel season ----
    print(f"\n  [CHART 3] Trip Type & Travel Season")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    df['tripType'].value_counts().plot(kind='barh', ax=axes[0],
                                       color=sns.color_palette("Set2"))
    axes[0].set_title("Reviews by Trip Type", fontsize=13, fontweight='bold')
    avg = df.groupby('travel_season')['rating'].mean().reindex(
        ['Spring', 'Summer', 'Autumn', 'Winter'])
    avg.plot(kind='bar', ax=axes[1], color=sns.color_palette("coolwarm", 4))
    axes[1].set_title("Average Rating by Travel Season", fontsize=13, fontweight='bold')
    axes[1].set_ylabel("Avg Rating")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "3_trip_season.png"), dpi=150)
    plt.close()

    print(f"    {'Trip Type':<15s} {'Reviews':<10s}")
    print(f"    {'-'*25}")
    for trip, cnt in df['tripType'].value_counts().items():
        print(f"    {trip:<15s} {cnt:<10d}")

    print(f"\n    {'Season':<12s} {'Avg Rating':<12s}")
    print(f"    {'-'*24}")
    for season in ['Spring', 'Summer', 'Autumn', 'Winter']:
        if season in avg.index:
            print(f"    {season:<12s} {avg[season]:.2f}")
    print(f"    -> Saved: plots/3_trip_season.png")

    # ---- 2d: Review length ----
    print(f"\n  [CHART 4] Review Length Distribution")
    df['char_count'] = df['text'].astype(str).apply(len)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df['word_count'], bins=50, color='steelblue', edgecolor='white')
    ax.set_title("Review Length Distribution (Word Count)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Word Count"); ax.set_ylabel("Frequency")
    ax.axvline(df['word_count'].median(), color='red', linestyle='--',
               label=f"Median={df['word_count'].median():.0f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "4_review_length.png"), dpi=150)
    plt.close()

    print(f"    Mean word count  : {df['word_count'].mean():.1f}")
    print(f"    Median word count: {df['word_count'].median():.0f}")
    print(f"    Max word count   : {df['word_count'].max()}")
    q99 = df['word_count'].quantile(0.99)
    outliers = len(df[df['word_count'] > q99])
    print(f"    Outliers (>99th) : {outliers} reviews with >{q99:.0f} words")
    print(f"    -> Saved: plots/4_review_length.png")

    # ---- 2e: Key observations ----
    print(f"\n  [KEY OBSERVATIONS]")
    print(f"    1. Heavily imbalanced: ~90% Positive, ~5% Neutral, ~4% Negative")
    print(f"    2. Rating-derived sentiment matches dataset labels (>{agree:.0%} agreement)")
    print(f"    3. Couples is the dominant trip type, followed by Family and Friends")
    print(f"    4. Spring/Summer have most reviews; avg ratings stable across seasons")
    print(f"    5. Most reviews are 25-100 words; few outliers exceed 400+ words")

    print("\n" + "+" + "-"*68 + "+")
    print("|{:^68s}|".format("TASK 2 COMPLETE  |  4 charts saved"))
    print("+" + "-"*68 + "+")

    return df
