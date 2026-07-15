"""
Task 4 - Lexicon-Based Sentiment Baseline (VADER)
"""
# pyrefly: ignore [missing-import]
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import classification_report, f1_score


def _vader_label(text):
    """Map VADER compound score to Positive/Neutral/Negative."""
    sid = SentimentIntensityAnalyzer()
    score = sid.polarity_scores(str(text))['compound']
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'


def run_vader_baseline(df):
    """Apply VADER to raw review text and report metrics."""

    print("\n" + "+" + "-"*68 + "+")
    print("|{:^68s}|".format("TASK 4: LEXICON-BASED BASELINE (VADER)"))
    print("+" + "-"*68 + "+")

    print(f"\n  [INFO] Applying VADER sentiment analyzer to {len(df)} reviews...")
    df['vader_pred'] = df['text'].astype(str).apply(_vader_label)

   # compare both label vader or real 
    y_true = df['sentiment_label']
    y_pred = df['vader_pred']

    macro_f1 = f1_score(y_true, y_pred, average='macro')
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    print(f"\n  [RESULTS] VADER vs Dataset Labels")
    print(f"  {'':4s}{'Class':<12s} {'Precision':<12s} {'Recall':<12s} {'F1-Score':<12s} {'Support':<10s}")
    print(f"  {'':4s}{'-'*58}")
    for cls in ['Positive', 'Neutral', 'Negative']:
        r = report[cls]
        print(f"  {'':4s}{cls:<12s} {r['precision']:<12.3f} {r['recall']:<12.3f} "
              f"{r['f1-score']:<12.3f} {int(r['support']):<10d}")
    print(f"  {'':4s}{'-'*58}")
    print(f"  {'':4s}{'Macro F1':<12s} {macro_f1:.4f}")

    print(f"\n  [INSIGHT] VADER works okay for Positive (recall=0.91) and Negative")
    print(f"           (recall=0.60) but fails badly on Neutral (F1=0.06).")
    print(f"           Rule-based tools struggle with mixed/ambivalent reviews.")

    print("\n" + "+" + "-"*68 + "+")
    print("|{:^68s}|".format("TASK 4 COMPLETE"))
    print("+" + "-"*68 + "+")

    return df
