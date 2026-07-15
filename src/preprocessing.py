"""
Task 3 - Text Preprocessing
"""
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Keep negation words - they flip sentiment polarity
NEGATION_WORDS = {
    'not', 'no', 'never', 'neither', 'nobody', 'nothing',
    'nowhere', 'nor', 'cannot', 'cant', "can't", "won't",
    "wouldn't", "shouldn't", "isn't", "aren't", "wasn't",
    "weren't", "don't", "doesn't", "didn't", "hasn't",
    "haven't", "hadn't"
}

stop_words = set(stopwords.words('english')) - NEGATION_WORDS
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    """Clean a single review: lowercase, remove punct/numbers, lemmatize."""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens
              if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)

    # lematization is to convert the word into their root form running -> run....


def preprocess(df):
    """Apply text cleaning pipeline to entire DataFrame."""

    print("\n" + "+" + "-"*68 + "+")
    print("|{:^68s}|".format("TASK 3: TEXT PREPROCESSING"))
    print("+" + "-"*68 + "+")

    print(f"\n  [PIPELINE]")
    print(f"    Step 1: Lowercasing")
    print(f"    Step 2: Remove URLs")
    print(f"    Step 3: Remove punctuation & numbers")
    print(f"    Step 4: Tokenization (NLTK word_tokenize)")
    print(f"    Step 5: Stopword removal (KEEP negation words)")
    print(f"    Step 6: Lemmatization (WordNetLemmatizer)")

    print(f"\n  [WHY KEEP NEGATION WORDS?]")
    print(f"    Words like 'not', 'never', 'don't' flip sentiment polarity.")
    print(f"    Removing them turns 'not good' -> 'good' = wrong sentiment!")

    print(f"\n  [BEFORE / AFTER EXAMPLES]")
    print(f"  " + "-"*66)

    sample_idx = [0, 40, 173]
    for i in sample_idx:
        if i < len(df):
            raw = str(df.loc[i, 'text'])[:100].replace('\n', ' ')
            cleaned = clean_text(raw)[:100]
            label = df.loc[i, 'sentiment_label']
            print(f"  Row {i} [{label}]:")
            print(f"    BEFORE: \"{raw}...\"")
            print(f"    AFTER : \"{cleaned}...\"")
            print()

    print(f"  [PROCESSING] Cleaning {len(df)} reviews...", end=" ")
    df['clean_text'] = df['text'].astype(str).apply(clean_text)
    print(f"Done!")

    print("\n" + "+" + "-"*68 + "+")
    print("|{:^68s}|".format("TASK 3 COMPLETE"))
    print("+" + "-"*68 + "+")

    return df
