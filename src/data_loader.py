"""
Task 1 - Data Loading & Schema Audit
"""
import pandas as pd


def load_and_audit(csv_path):
    """Load CSV, print schema audit, drop duplicates, return clean DataFrame."""

    print("\n" + "+" + "-"*68 + "+")
    print("|{:^68s}|".format("TASK 1: DATA LOADING & SCHEMA AUDIT"))
    print("+" + "-"*68 + "+")

    df = pd.read_csv(csv_path)

    print(f"\n  [INFO] Dataset loaded successfully!")
    # tell how many rows and columns 
    print(f"  [INFO] Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    print(f"\n  {'Column Name':<25s} {'Dtype':<15s} {'Non-Null':<10s} {'Nulls':<8s}")
    print("  " + "-"*58)
    #loop to how how many rows ha contain data and how manhy empty
    for col in df.columns:
        dtype = str(df[col].dtype) # check data is in number or text
        non_null = df[col].notna().sum() # count how many rows are fill with data or how many empty
        nulls = df[col].isna().sum() # kitni rows aisi hain jis me data missing ha 
        print(f"  {col:<25s} {dtype:<15s} {non_null:<10d} {nulls:<8d}")

    print(f"\n  [KEY COLUMNS]")
    print(f"    Review text     : 'text'")
    print(f"    Star rating     : 'rating'  (1-5 scale)")
    print(f"    Sentiment label : 'sentiment_label'  (Positive / Neutral / Negative)")

    print(f"\n  [SAMPLE REVIEWS]")
    for i in range(3):
        txt = str(df.loc[i, 'text'])[:100].replace('\n', ' ')
        print(f"    Row {i}: [{df.loc[i,'rating']}*] \"{txt}...\"")

    dup_count = df.duplicated(subset=['text']).sum() 
    print(f"\n  [DUPLICATES] Found {dup_count} duplicate reviews -> Removing them...")

    # drop duplicates rows  and then set the rows number from 0
    df = df.drop_duplicates(subset=['text']).reset_index(drop=True)
    print(f"  [RESULT]  Clean dataset: {df.shape[0]} rows x {df.shape[1]} columns")

    print("\n" + "+" + "-"*68 + "+")
    print("|{:^68s}|".format("TASK 1 COMPLETE"))
    print("+" + "-"*68 + "+")
# return clean data after dropping duplicates
    return df
