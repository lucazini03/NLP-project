import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from config import *


def load_data():
    """Load the CSV file"""
    df = pd.read_csv(
        DATA_FILE,
        engine="python",
        sep=",",
        quotechar="\"",
        escapechar="\\",
        on_bad_lines="skip"
    )

    # Rename columns
    df = df.rename(columns={
        "post": "text",
        "female": "label",
        "auhtor_ID": "author_id"
    })

    df = df[["text", "label", "author_id"]].dropna()

    print(f"Loaded {len(df)} posts")
    print(f"Label distribution:\n{df.label.value_counts()}")

    return df


def split_by_author(df):
    """Split data ensuring no author overlap"""
    gss = GroupShuffleSplit(test_size=TEST_SIZE, n_splits=1, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(df, groups=df["author_id"]))

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    # Add length feature
    train_df["length"] = train_df["text"].str.len()
    test_df["length"] = test_df["text"].str.len()

    print(f"\nTrain: {len(train_df)} posts")
    print(f"Test: {len(test_df)} posts")
    print(f"Author overlap: {len(set(train_df['author_id']) & set(test_df['author_id']))} (should be 0)")

    return train_df, test_df


def remove_near_duplicates(train_df, test_df):
    """Remove posts from test set that are too similar to training - FAST VERSION"""
    print("\nRemoving near-duplicates...")

    # Speed optimization 1: Use smaller ngram range (faster)
    # Speed optimization 2: Higher min_df (fewer features = faster)
    vectorizer = TfidfVectorizer(ngram_range=(4, 4), min_df=5, max_features=10000)

    # Speed optimization 3: Sample train set if it's huge
    if len(train_df) > 50000:
        train_sample = train_df.sample(50000, random_state=42)
        print(f"  Using sample of 50k training posts for speed...")
    else:
        train_sample = train_df

    train_vecs = vectorizer.fit_transform(train_sample["text"])
    test_vecs = vectorizer.transform(test_df["text"])

    similarities = cosine_similarity(test_vecs, train_vecs)
    max_sim = similarities.max(axis=1)

    # Keep only posts with <90% similarity
    keep_mask = max_sim < 0.9
    clean_test_df = test_df[keep_mask].copy()

    print(f"  Removed {(~keep_mask).sum()} near-duplicates ({100 * (~keep_mask).sum() / len(test_df):.1f}%)")
    print(f"  Clean test set: {len(clean_test_df)} posts")

    return clean_test_df