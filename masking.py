"""
masking.py – Counterfactual text generation via pure token deletion.

Generates 4 text conditions from a single spaCy POS pass per document:
  1. Original        – cleaned text, untouched.
  2. Structural      – delete tokens whose lemma ∈ STRUCTURAL_GENDER_TERMS.
  3. Random (control) – delete k random non-gender, non-punct tokens
                        (k = # tokens deleted in structural), seeded.
  4. Topical         – delete structural ∪ {NOUN, PROPN} tokens.

All masking is *pure deletion* (no placeholder strings).
Double spaces from deletion are collapsed.
"""
import re
import logging
from collections import defaultdict

import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm

from config import STRUCTURAL_GENDER_TERMS, RANDOM_STATE, get_rng

logger = logging.getLogger(__name__)

# Load spaCy once
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# Collapse multiple spaces after deletion
_MULTI_SPACE = re.compile(r" {2,}")


def _collapse_spaces(text: str) -> str:
    return _MULTI_SPACE.sub(" ", text).strip()


def _build_text_from_mask(doc, delete_indices: set[int]) -> str:
    """Reconstruct text from a spaCy Doc, skipping deleted token indices."""
    tokens = []
    for i, tok in enumerate(doc):
        if i not in delete_indices:
            tokens.append(tok.text_with_ws)
    return _collapse_spaces("".join(tokens))


def generate_masked_versions(
    texts: list[str],
    rng: np.random.Generator | None = None,
    batch_size: int = 256,
) -> dict[str, list[str]]:
    """Generate all 4 text versions in a single spaCy pass.

    Parameters
    ----------
    texts : list[str]
        Cleaned original texts.
    rng : np.random.Generator, optional
        Seeded RNG for random masking.  Falls back to config.get_rng().
    batch_size : int
        spaCy pipe batch size.

    Returns
    -------
    dict with keys "original", "structural", "random", "topical",
    each mapping to a list[str] of the same length as *texts*.
    """
    if rng is None:
        rng = get_rng()

    originals: list[str] = []
    structurals: list[str] = []
    randoms: list[str] = []
    topicals: list[str] = []

    for doc in tqdm(
        nlp.pipe(texts, batch_size=batch_size),
        total=len(texts),
        desc="Generating 4 text conditions (single spaCy pass)",
    ):
        # ── Identify deletion sets on the ORIGINAL doc ──

        # Structural: tokens whose lemma is in the gender-term set
        structural_del: set[int] = set()
        for i, tok in enumerate(doc):
            if tok.lemma_.lower() in STRUCTURAL_GENDER_TERMS:
                structural_del.add(i)

        # Topical: structural ∪ {NOUN, PROPN}
        topical_del: set[int] = set()
        for i, tok in enumerate(doc):
            if i in structural_del or tok.pos_ in ("NOUN", "PROPN"):
                topical_del.add(i)

        # Random (control): k random non-gender, non-punct/space tokens
        k = len(structural_del)
        if k > 0:
            eligible = [
                i for i, tok in enumerate(doc)
                if (i not in structural_del
                    and tok.pos_ not in ("PUNCT", "SPACE")
                    and not tok.is_space)
            ]
            if len(eligible) <= k:
                random_del = set(eligible)
            else:
                chosen = rng.choice(eligible, size=k, replace=False)
                random_del = set(chosen.tolist())
        else:
            random_del = set()  # no-op when k = 0

        # ── Build text versions ──
        originals.append(doc.text)  # already cleaned
        structurals.append(_build_text_from_mask(doc, structural_del))
        randoms.append(_build_text_from_mask(doc, random_del))
        topicals.append(_build_text_from_mask(doc, topical_del))

    return {
        "original": originals,
        "structural": structurals,
        "random": randoms,
        "topical": topicals,
    }


# ───────────── Apply masking to a DataFrame ───────────────────

def apply_masking(df: pd.DataFrame, rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Add text_structural, text_random, text_topical columns in-place.

    The ``text`` column is assumed to already be cleaned.
    Returns the modified DataFrame.
    """
    if rng is None:
        rng = get_rng()

    versions = generate_masked_versions(df["text"].tolist(), rng=rng)
    df["text_structural"] = versions["structural"]
    df["text_random"] = versions["random"]
    df["text_topical"] = versions["topical"]

    return df


def apply_masking_in_chunks(df: pd.DataFrame, num_chunks: int = 5, cache_dir: str = ".", rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Apply masking in chunks and save intermediate results to avoid losing progress.
    
    Splits the DataFrame into `num_chunks`, processes them iteratively, and saves 
    each chunk. If a chunk already exists in `cache_dir`, it is loaded from disk.
    """
    import os
    import pickle
    import numpy as np
    
    if rng is None:
        rng = get_rng()
        
    os.makedirs(cache_dir, exist_ok=True)
    chunks = np.array_split(df, num_chunks)
    processed_chunks = []
    
    for i, chunk in enumerate(chunks):
        chunk_file = os.path.join(cache_dir, f"masked_chunk_{i}.pkl")
        if os.path.exists(chunk_file):
            logger.info(f"Loading chunk {i+1}/{num_chunks} from {chunk_file}...")
            print(f"Loading chunk {i+1}/{num_chunks} from {chunk_file}...")
            with open(chunk_file, "rb") as f:
                processed_chunk = pickle.load(f)
        else:
            logger.info(f"Processing chunk {i+1}/{num_chunks}...")
            print(f"Processing chunk {i+1}/{num_chunks}...")
            processed_chunk = apply_masking(chunk.copy(), rng=rng)
            with open(chunk_file, "wb") as f:
                pickle.dump(processed_chunk, f)
            print(f"Saved chunk {i+1}/{num_chunks} to {chunk_file}")
            
        processed_chunks.append(processed_chunk)
        
    return pd.concat(processed_chunks, ignore_index=False)


# ───────────── Masking statistics (Sparsity) ──────────────────

def compute_masking_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-class masking sparsity statistics.

    For each of the 4 text conditions, report mean/median tokens deleted
    per document, broken down by class (0 = Male, 1 = Female).
    """
    label_map = {0: "Male", 1: "Female"}
    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing masking stats"):
        orig_tokens = row["text"].split()
        n_orig = len(orig_tokens)
        for cond in ("structural", "random", "topical"):
            col = f"text_{cond}"
            masked_tokens = row[col].split()
            n_deleted = n_orig - len(masked_tokens)
            rows.append({
                "class": label_map.get(row["label"], row["label"]),
                "condition": cond,
                "tokens_deleted": n_deleted,
                "tokens_original": n_orig,
                "deletion_ratio": n_deleted / n_orig if n_orig > 0 else 0.0,
            })

    stats_df = pd.DataFrame(rows)

    # Aggregate
    agg = (
        stats_df
        .groupby(["condition", "class"])
        .agg(
            mean_deleted=("tokens_deleted", "mean"),
            median_deleted=("tokens_deleted", "median"),
            std_deleted=("tokens_deleted", "std"),
            mean_ratio=("deletion_ratio", "mean"),
        )
        .reset_index()
    )

    logger.info("Masking sparsity statistics:\n%s", agg.to_string(index=False))
    print("\n=== Masking Sparsity Statistics ===")
    print(agg.to_string(index=False))

    return agg