"""
prepare_data.py – Data loading, cleaning, filtering, and fold generation.

Implements:
  1. clean_text(): HTML entity decoding, zero-width/control char stripping,
     Unicode NFKC normalisation, whitespace collapsing.
  2. Placeholder / deleted-post filtering with per-step logging.
  3. Author-disjoint near-duplicate removal (applied to cleaned text).
  4. StratifiedGroupKFold generation (author-disjoint, seeded).
"""
import html
import re
import unicodedata
import logging

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import DATA_FILE, RANDOM_STATE, N_FOLDS

logger = logging.getLogger(__name__)

# ───────────────────── Text cleaning ──────────────────────────

# Regex for zero-width / invisible Unicode chars
_ZWSP_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")
# Collapse runs of whitespace (including newlines) to a single space
_WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Clean a single text string.

    Steps (in order):
      1. Decode HTML entities  (e.g. &amp; -> &)
      2. Strip zero-width / BOM characters
      3. Unicode NFKC normalisation
      4. Collapse extra whitespace and strip leading/trailing space
    """
    text = html.unescape(text)
    text = _ZWSP_RE.sub("", text)
    text = unicodedata.normalize("NFKC", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


# ───────────────────── Data loading ───────────────────────────

_PLACEHOLDER_STRINGS = {"[deleted]", "[removed]"}


def load_and_clean_data() -> pd.DataFrame:
    """Load CSV, clean text, filter placeholders, and log drop counts."""
    df = pd.read_csv(
        DATA_FILE,
        engine="python",
        sep=",",
        quotechar='"',
        escapechar="\\",
        on_bad_lines="skip",
    )

    # Standardise column names
    df = df.rename(columns={
        "post": "text",
        "female": "label",
        "auhtor_ID": "author_id",
    })
    df = df[["text", "label", "author_id"]].dropna()
    n_after_load = len(df)
    logger.info("Loaded %d posts from CSV.", n_after_load)

    # ── Step 1: clean text ──
    df["text"] = df["text"].astype(str).map(clean_text)
    logger.info("Applied clean_text() to all posts.")

    # ── Step 2: filter placeholders and empties ──
    n_before = len(df)
    mask_placeholder = df["text"].str.strip().isin(_PLACEHOLDER_STRINGS)
    n_placeholder = mask_placeholder.sum()
    df = df[~mask_placeholder]
    logger.info("Dropped %d placeholder posts ([deleted]/[removed]).", n_placeholder)

    mask_empty = df["text"].str.strip().eq("")
    n_empty = mask_empty.sum()
    df = df[~mask_empty]
    logger.info("Dropped %d empty posts after cleaning.", n_empty)

    n_total_dropped = n_before - len(df)
    logger.info(
        "Total rows dropped in filtering: %d  (remaining: %d)",
        n_total_dropped, len(df),
    )

    # Reset index
    df = df.reset_index(drop=True)

    # Summary
    logger.info("Label distribution:\n%s", df["label"].value_counts().to_string())
    print(f"Loaded and cleaned {len(df)} posts  "
          f"(dropped {n_placeholder} placeholder, {n_empty} empty)")
    print(f"Label distribution:\n{df['label'].value_counts()}")

    return df


# ───────────────── Near-duplicate removal ─────────────────────

def remove_near_duplicates(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    threshold: float = 0.95,
    chunk_size: int = 5000,
) -> np.ndarray:
    """Remove near-duplicate test indices that are too similar to any train row.

    Works on the cleaned ``text`` column via TF-IDF cosine similarity.
    Returns a *filtered* array of test indices.
    """
    train_texts = df.loc[train_idx, "text"].astype(str)
    test_texts = df.loc[test_idx, "text"].astype(str)

    tfidf = TfidfVectorizer(stop_words="english", min_df=2)
    train_tfidf = tfidf.fit_transform(train_texts)
    test_tfidf = tfidf.transform(test_texts)

    to_remove: set[int] = set()
    num_test = test_tfidf.shape[0]

    for i in range(0, num_test, chunk_size):
        end_i = min(i + chunk_size, num_test)
        sim = cosine_similarity(test_tfidf[i:end_i], train_tfidf)
        for chunk_idx in range(sim.shape[0]):
            if np.any(sim[chunk_idx] > threshold):
                to_remove.add(i + chunk_idx)

    # Map positional offsets back to actual index values
    test_idx_list = list(test_idx)
    filtered = np.array(
        [idx for pos, idx in enumerate(test_idx_list) if pos not in to_remove]
    )

    logger.info(
        "Near-duplicate removal: dropped %d / %d test docs (threshold=%.2f).",
        len(to_remove), num_test, threshold,
    )
    print(f"Near-duplicate removal: dropped {len(to_remove)} test docs.")
    return filtered


# ───────────── StratifiedGroupKFold generation ────────────────

def generate_folds(
    df: pd.DataFrame,
    dedup_threshold: float = 0.95,
) -> list[dict]:
    """Generate K folds with author-disjoint, stratified splits.

    Each fold dict contains:
        {"fold": int, "train_idx": np.ndarray, "test_idx": np.ndarray}

    Near-duplicate removal is applied to each fold's test indices.
    """
    sgkf = StratifiedGroupKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )

    folds = []
    for fold_i, (train_idx, test_idx) in enumerate(
        sgkf.split(df, y=df["label"], groups=df["author_id"])
    ):
        # Verify no author overlap
        train_authors = set(df.loc[train_idx, "author_id"])
        test_authors = set(df.loc[test_idx, "author_id"])
        overlap = train_authors & test_authors
        assert len(overlap) == 0, (
            f"Fold {fold_i}: author overlap detected ({len(overlap)} authors)!"
        )

        # Near-duplicate removal on test set
        test_idx = remove_near_duplicates(
            df, train_idx, test_idx, threshold=dedup_threshold
        )

        folds.append({
            "fold": fold_i,
            "train_idx": train_idx,
            "test_idx": test_idx,
        })
        logger.info(
            "Fold %d: train=%d, test=%d (after dedup)",
            fold_i, len(train_idx), len(test_idx),
        )
        print(f"Fold {fold_i}: train={len(train_idx)}, test={len(test_idx)}")

    return folds