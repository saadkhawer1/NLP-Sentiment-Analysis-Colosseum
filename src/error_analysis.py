"""
Task 6 - Error Analysis
"""
import pandas as pd


def run_error_analysis(df, model, tfidf, X_test, y_test, X_test_tfidf):
    """Extract and analyze misclassified reviews from the best model."""

    print("\n" + "+" + "-"*68 + "+")
    print("|{:^68s}|".format("TASK 6: ERROR ANALYSIS"))
    print("+" + "-"*68 + "+")


#test data and check it was predicted correctly ya incorrectly  
    y_pred = model.predict(X_test_tfidf)
    # make true those review which are predicted in correctly
    mask = y_pred != y_test.values


  # mistake new table 
    misclassified = pd.DataFrame({
        'text': X_test.values[mask],  # original review
        'true_label': y_test.values[mask], # true label
        'pred_label': y_pred[mask] # predicted label
    })

    total = len(y_test)
    wrong = len(misclassified)
    acc = (total - wrong) / total * 100

    print(f"\n  [SUMMARY]")
    print(f"    Total test reviews : {total}")
    print(f"    Correctly classified: {total - wrong}")
    print(f"    Misclassified       : {wrong}")
    print(f"    Test accuracy       : {acc:.1f}%")

    n_show = min(10, wrong)
    print(f"\n  [MISCLASSIFIED EXAMPLES] (showing {n_show})")
    print(f"  {'-'*66}")

 # print first 10 wrong predictions
    for i in range(n_show):
        row = misclassified.iloc[i]
        txt = str(row['text'])[:120].replace('\n', ' ')
        print(f"\n  [{i+1}] True: {row['true_label']:<10s}  Predicted: {row['pred_label']}")
        print(f"      \"{txt}...\"")

    print(f"\n  {'-'*66}")
    print(f"\n  [ERROR PATTERNS IDENTIFIED]")
    print(f"    1. MIXED SENTIMENT  : Praises Colosseum but complains about")
    print(f"                          queues/scammers -> model misses complaints")
    print(f"    2. SHORT REVIEWS    : Too few words for reliable classification")
    print(f"    3. SARCASM / IRONY  : 'amazing wait times' reads as positive")
    print(f"    4. NEUTRAL AMBIGUITY: 3-star reviews with mixed language")
    print(f"    5. DOMAIN COMPLAINTS: Queue/heat/ticket/scam words are too")
    print(f"                          neutral for the model to flag as negative")

    print("\n" + "+" + "-"*68 + "+")
    print("|{:^68s}|".format("TASK 6 COMPLETE"))
    print("+" + "-"*68 + "+")
