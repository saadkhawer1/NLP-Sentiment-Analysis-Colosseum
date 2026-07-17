"""
=============================================================================
  Sentiment Analysis - Rome Colosseum Visitor Reviews (2019-2026)
  Main orchestrator: runs all task modules (Tasks 1-6)
=============================================================================
"""
import os
import warnings
import joblib

warnings.filterwarnings('ignore')

# Paths
BASE = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE, "rome_colosseum_visitor_reviews_final.csv")
PLOTS_DIR = os.path.join(BASE, "plots")

# Import task modules
from src.data_loader import load_and_audit
from src.eda import run_eda
from src.preprocessing import preprocess
from src.lexicon_baseline import run_vader_baseline
from src.ml_classifier import run_ml_classifiers
from src.error_analysis import run_error_analysis


def main():
    print()
    print("+" + "="*68 + "+")
    print("|{:^68s}|".format(""))
    print("|{:^68s}|".format("SENTIMENT ANALYSIS PIPELINE"))
    print("|{:^68s}|".format("Rome Colosseum Visitor Reviews (2019-2026)"))
    print("|{:^68s}|".format(""))
    print("+" + "="*68 + "+")

    # Task 1
    df = load_and_audit(CSV_FILE)

    # Task 2
    df = run_eda(df, PLOTS_DIR)

    # Task 3
    df = preprocess(df)

    # Task 4
    df = run_vader_baseline(df)

    # Task 5
    df, best_model, tfidf, X_test, y_test, X_test_tfidf, results = \
        run_ml_classifiers(df, PLOTS_DIR)

    # Task 6
    run_error_analysis(df, best_model, tfidf, X_test, y_test, X_test_tfidf)

    # Save model and vectorizer for the Web App
    model_path = os.path.join(BASE, "sentiment_model.pkl")
    vectorizer_path = os.path.join(BASE, "tfidf_vectorizer.pkl")
    joblib.dump(best_model, model_path)
    joblib.dump(tfidf, vectorizer_path)
    print(f"\n  [INFO] Saved Model to {model_path}")
    print(f"  [INFO] Saved Vectorizer to {vectorizer_path}")

    # Done
    print()
    print("+" + "="*68 + "+")
    print("|{:^68s}|".format("ALL 6 TASKS COMPLETE"))
    print("|{:^68s}|".format("5 charts saved in ./plots/"))
    print("|{:^68s}|".format("Report available: report.md"))
    print("+" + "="*68 + "+")
    print()


if __name__ == "__main__":
    main()
