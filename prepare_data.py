import pandas as pd
import numpy as np
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


def remove_near_duplicates(train_df, test_df, threshold=0.95, chunk_size=5000):
    """Remove near-duplicate test rows that are too similar to any train row.

    Compares TF-IDF cosine similarity between `text` columns of train and test.
    Returns a filtered `test_df` with near-duplicates removed.
    """
    # Ensure expected text column exists
    text_col = 'text' if 'text' in train_df.columns else ('post' if 'post' in train_df.columns else None)
    if text_col is None or text_col not in test_df.columns:
        raise KeyError('No text/post column found in dataframes')

    print(f"Starting duplicate removal: train {len(train_df)} rows, test {len(test_df)} rows...")
    tfidf = TfidfVectorizer(stop_words='english', min_df=2)

    # Fit on train texts and transform both sets
    train_tfidf = tfidf.fit_transform(train_df[text_col].astype(str))
    test_tfidf = tfidf.transform(test_df[text_col].astype(str))

    to_remove = set()
    num_test = test_tfidf.shape[0]

    for i in range(0, num_test, chunk_size):
        end_i = min(i + chunk_size, num_test)
        # compute similarity of this test chunk against all train rows
        sim_matrix = cosine_similarity(test_tfidf[i:end_i], train_tfidf)

        for chunk_idx, row in enumerate(sim_matrix):
            actual_idx = i + chunk_idx
            # find any train index with similarity > threshold
            if np.any(row > threshold):
                to_remove.add(actual_idx)

        print(f"Processed test rows {end_i}/{num_test}...")

    if to_remove:
        test_cleaned = test_df.drop(test_df.index[list(to_remove)]).reset_index(drop=True)
    else:
        test_cleaned = test_df.reset_index(drop=True)

    print(f"Removed {len(to_remove)} near-duplicates from test. New test size: {len(test_cleaned)}")
    return test_cleaned