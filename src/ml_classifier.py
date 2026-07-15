"""
Task 5 - Machine Learning Classifiers & Class Imbalance Handling
"""
import os
from collections import Counter

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, cohen_kappa_score,
                             balanced_accuracy_score)

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


def _eval_model(name, y_true, y_pred, results):
    """Evaluate and store results for one model."""
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    print(f"\n    {'Class':<12s} {'Prec':<8s} {'Recall':<8s} {'F1':<8s}")
    print(f"    {'-'*36}")
    for cls in ['Positive', 'Neutral', 'Negative']:
        r = report.get(cls, {})
        print(f"    {cls:<12s} {r.get('precision',0):.3f}   "
              f"{r.get('recall',0):.3f}   {r.get('f1-score',0):.3f}")
    print(f"    {'-'*36}")
    print(f"    Macro F1: {macro_f1:.4f}  |  Kappa: {kappa:.4f}  |  Bal.Acc: {bal_acc:.4f}")

    entry = {'Model': name, 'Macro_F1': round(macro_f1, 4),
             'Kappa': round(kappa, 4), 'Bal_Acc': round(bal_acc, 4)}
    for cls in ['Positive', 'Neutral', 'Negative']:
        entry[f'{cls}_P'] = round(report.get(cls, {}).get('precision', 0), 3)
        entry[f'{cls}_R'] = round(report.get(cls, {}).get('recall', 0), 3)
    results.append(entry)
    return results


def run_ml_classifiers(df, plots_dir):
    """Train baseline + imbalance-handled classifiers."""

    print("\n" + "+" + "-"*68 + "+")
    print("|{:^68s}|".format("TASK 5: ML CLASSIFIERS & CLASS IMBALANCE"))
    print("+" + "-"*68 + "+")

    X = df['clean_text']
    y = df['sentiment_label']

    # ---- Class distribution ----
    print(f"\n  [CLASS DISTRIBUTION]")
    print(f"    {'Class':<12s} {'Count':<8s} {'Percentage':<12s} {'Bar'}")
    print(f"    {'-'*50}")
    for cls in ['Positive', 'Neutral', 'Negative']:
        cnt = (y == cls).sum()
        pct = cnt / len(y) * 100
        bar = '#' * int(pct / 2)
        print(f"    {cls:<12s} {cnt:<8d} {pct:>5.1f}%       {bar}")

    print(f"\n  [WHY ACCURACY IS MISLEADING]")
    print(f"    A dummy 'always Positive' classifier gets ~90% accuracy")
    print(f"    but has 0% recall on Negative/Neutral = totally useless!")
    print(f"    We use Macro F1 as primary metric (treats all classes equally).")

    # ---- Split & vectorize ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

        
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    print(f"\n  [SPLIT] Train: {X_train_tfidf.shape[0]} | Test: {X_test_tfidf.shape[0]}")
    print(f"  [TF-IDF] {X_train_tfidf.shape[1]} features (unigrams + bigrams)")

    results = []

    # ======== BASELINES ========
    print(f"\n  {'='*60}")
    print(f"  BASELINE MODELS (no imbalance handling)")
    print(f"  {'='*60}")

    print(f"\n  [1] Logistic Regression (baseline)")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_tfidf, y_train)
    results = _eval_model("LR (baseline)", y_test, lr.predict(X_test_tfidf), results)

    print(f"\n  [2] Multinomial Naive Bayes (baseline)")
    nb = MultinomialNB()
    nb.fit(X_train_tfidf, y_train)
    results = _eval_model("MNB (baseline)", y_test, nb.predict(X_test_tfidf), results)

    # ======== TECHNIQUE 1 ========
    print(f"\n  {'='*60}")
    print(f"  TECHNIQUE 1: Class Weighting (class_weight='balanced')")
    print(f"  {'='*60}")

    print(f"\n  [3] Logistic Regression (balanced)")
    lr_bal = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr_bal.fit(X_train_tfidf, y_train)
    results = _eval_model("LR (balanced)", y_test, lr_bal.predict(X_test_tfidf), results)

    print(f"\n  [4] LinearSVC (balanced)")
    svc = LinearSVC(class_weight='balanced', max_iter=2000, random_state=42)
    svc.fit(X_train_tfidf, y_train)
    results = _eval_model("SVC (balanced)", y_test, svc.predict(X_test_tfidf), results)

    # ======== TECHNIQUE 2 ========
    print(f"\n  {'='*60}")
    print(f"  TECHNIQUE 2: Random Under-sampling")
    print(f"  {'='*60}")

    rus = RandomUnderSampler(random_state=42)
    X_rus, y_rus = rus.fit_resample(X_train_tfidf, y_train)
    dist = Counter(y_rus)
    print(f"  Resampled: {dict(dist)}")

    print(f"\n  [5] Logistic Regression (undersampled)")
    lr_rus = LogisticRegression(max_iter=1000, random_state=42)
    lr_rus.fit(X_rus, y_rus)
    results = _eval_model("LR (undersample)", y_test, lr_rus.predict(X_test_tfidf), results)

    # ======== TECHNIQUE 3 ========
    print(f"\n  {'='*60}")
    print(f"  TECHNIQUE 3: Random Over-sampling")
    print(f"  {'='*60}")

    ros = RandomOverSampler(random_state=42)
    X_ros, y_ros = ros.fit_resample(X_train_tfidf, y_train)
    dist = Counter(y_ros)
    print(f"  Resampled: {dict(dist)}")

    print(f"\n  [6] Logistic Regression (oversampled)")
    lr_ros = LogisticRegression(max_iter=1000, random_state=42)
    lr_ros.fit(X_ros, y_ros)
    results = _eval_model("LR (oversample)", y_test, lr_ros.predict(X_test_tfidf), results)

    # ======== HYPERPARAMETER TUNING ========
    print(f"\n  {'='*60}")
    print(f"  HYPERPARAMETER TUNING (GridSearchCV)")
    print(f"  {'='*60}")

    param_grid = {'C': [0.1, 1, 10], 'solver': ['lbfgs', 'liblinear']}
    gs = GridSearchCV(
        LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        param_grid, scoring='f1_macro', cv=3, n_jobs=-1)
    gs.fit(X_train_tfidf, y_train)
    print(f"  Best params: {gs.best_params_}")
    print(f"  Best CV Macro F1: {gs.best_score_:.4f}")

    print(f"\n  [7] LR (tuned + balanced)")
    results = _eval_model("LR (tuned)", y_test, gs.predict(X_test_tfidf), results)

    # ======== FINAL COMPARISON TABLE ========
    print(f"\n  {'='*68}")
    print(f"  FINAL MODEL COMPARISON TABLE")
    print(f"  {'='*68}")
    print(f"  {'Model':<20s} {'MacroF1':<9s} {'Pos_R':<7s} {'Neu_R':<7s} {'Neg_R':<7s} {'Kappa':<7s}")
    print(f"  {'-'*57}")
    for r in results:
        print(f"  {r['Model']:<20s} {r['Macro_F1']:<9.4f} "
              f"{r['Positive_R']:<7.3f} {r['Neutral_R']:<7.3f} "
              f"{r['Negative_R']:<7.3f} {r['Kappa']:<7.4f}")
    print(f"  {'-'*57}")

    # Best model
    best_entry = max(results, key=lambda x: x['Macro_F1'])
    print(f"\n  [BEST MODEL] {best_entry['Model']} (Macro F1 = {best_entry['Macro_F1']:.4f})")

    # ---- Comparison plot ----
    fig, ax = plt.subplots(figsize=(10, 6))
    models = [r['Model'] for r in results]
    scores = [r['Macro_F1'] for r in results]
    colors = sns.color_palette("viridis", len(results))
    ax.barh(models, scores, color=colors)
    ax.set_xlabel("Macro F1 Score")
    ax.set_title("Model Comparison - Macro F1", fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    for i, v in enumerate(scores):
        ax.text(v + 0.01, i, f'{v:.3f}', va='center')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "5_model_comparison.png"), dpi=150)
    plt.close()
    print(f"  -> Saved: plots/5_model_comparison.png")

    print(f"\n  [RECOMMENDATION]")
    print(f"    For a tourism company wanting to flag negative reviews:")
    print(f"    -> Use LR with class_weight='balanced'")
    print(f"    -> Simplest approach, no data duplication, best minority recall")
    print(f"    -> Small cost in Positive precision is acceptable")

    print("\n" + "+" + "-"*68 + "+")
    print("|{:^68s}|".format("TASK 5 COMPLETE  |  1 chart saved"))
    print("+" + "-"*68 + "+")

    best_model = gs.best_estimator_
    return df, best_model, tfidf, X_test, y_test, X_test_tfidf, results
