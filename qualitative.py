"""
qualitative.py – Per-example feature-importance analysis across the 4-way ablation.

Updated to use the new column names (text_structural, text_random, text_topical)
and pure-deletion masking (no placeholder tokens).
"""
import os
import logging

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from config import (
    RANDOM_STATE, TEXT_CONDITIONS, TEXT_COLUMNS, RESULTS_DIR,
)

logger = logging.getLogger(__name__)


def get_top_features_for_text(
    text: str,
    vectorizer: TfidfVectorizer,
    clf: LogisticRegression,
    top_n: int = 5,
) -> list[tuple[str, float]]:
    """Return the top-N most influential TF-IDF features for a single text."""
    vec = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()
    coefs = clf.coef_[0]

    feature_indices = vec.nonzero()[1]
    feature_scores = [
        (feature_names[i], coefs[i] * vec[0, i])
        for i in feature_indices
    ]
    feature_scores.sort(key=lambda x: abs(x[1]), reverse=True)
    return feature_scores[:top_n]


def create_ablation_analysis_csv(
    test_df: pd.DataFrame,
    original_pipe: Pipeline,
    masked_pipe: Pipeline,
    masked_col: str,
    output_name: str,
    sample_size: int = 200,
    n_examples: int = 5,
) -> pd.DataFrame:
    """Compare original vs masked model predictions and save qualitative CSV.

    This function compares a model trained on *original* text against a
    model trained on the *masked* variant, both evaluated on their
    respective text versions.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"QUALITATIVE ANALYSIS: {output_name.upper()}")
    print(f"{'=' * 60}")

    if len(test_df) > sample_size:
        test_sample = test_df.sample(
            sample_size, random_state=RANDOM_STATE
        ).reset_index(drop=True)
    else:
        test_sample = test_df.reset_index(drop=True)

    orig_preds = original_pipe.predict(test_sample["text"])
    mask_preds = masked_pipe.predict(test_sample[masked_col])

    # predict_proba not available for LinearSVC — guard with hasattr
    has_proba_orig = hasattr(original_pipe, "predict_proba")
    has_proba_mask = hasattr(masked_pipe, "predict_proba")

    orig_proba = (
        original_pipe.predict_proba(test_sample["text"])
        if has_proba_orig else None
    )
    mask_proba = (
        masked_pipe.predict_proba(test_sample[masked_col])
        if has_proba_mask else None
    )

    vec_orig = original_pipe.named_steps["tfidf"]
    clf_orig = original_pipe.named_steps["clf"]
    vec_mask = masked_pipe.named_steps["tfidf"]
    clf_mask = masked_pipe.named_steps["clf"]

    rows = []
    for i, row in test_sample.iterrows():
        orig_feats = get_top_features_for_text(row["text"], vec_orig, clf_orig)
        mask_feats = get_top_features_for_text(row[masked_col], vec_mask, clf_mask)

        entry = {
            "original_text": row["text"][:200],
            "masked_text": row[masked_col][:200],
            "true_label": "Female" if row["label"] == 1 else "Male",
            "original_prediction": "Female" if orig_preds[i] == 1 else "Male",
            "original_correct": bool(orig_preds[i] == row["label"]),
            "original_top_features": ", ".join(
                f"{f} ({w:+.3f})" for f, w in orig_feats
            ),
            "masked_prediction": "Female" if mask_preds[i] == 1 else "Male",
            "masked_correct": bool(mask_preds[i] == row["label"]),
            "masked_top_features": ", ".join(
                f"{f} ({w:+.3f})" for f, w in mask_feats
            ),
            "prediction_changed": bool(orig_preds[i] != mask_preds[i]),
        }

        if orig_proba is not None:
            entry["original_confidence"] = f"{orig_proba[i][orig_preds[i]]:.3f}"
        if mask_proba is not None:
            entry["masked_confidence"] = f"{mask_proba[i][mask_preds[i]]:.3f}"

        rows.append(entry)

    result_df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, f"{output_name}_analysis.csv")
    result_df.to_csv(csv_path, index=False)

    # Summary
    print(f"  Original accuracy: {result_df['original_correct'].mean():.2%}")
    print(f"  Masked accuracy:   {result_df['masked_correct'].mean():.2%}")
    print(f"  Predictions changed: {result_df['prediction_changed'].mean():.1%}")

    # Show example flips
    flipped = result_df[result_df["prediction_changed"]].head(n_examples)
    if len(flipped) == 0:
        print("  No prediction changes found in sample.")
    else:
        print(f"\n  Example flips ({output_name}):")
        for idx, frow in flipped.iterrows():
            print(f"    Case {idx}: {frow['original_prediction']} → "
                  f"{frow['masked_prediction']} (true: {frow['true_label']})")
            print(f"      Orig features: {frow['original_top_features']}")
            print(f"      Mask features: {frow['masked_top_features']}")

    return result_df


def analyze_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Train LR on each text condition and run qualitative comparison.

    Trains 4 pipelines (original, structural, random, topical) and compares
    each masked pipeline against the original pipeline.

    Returns a dict mapping condition name → analysis DataFrame.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n" + "=" * 80)
    print("QUALITATIVE FEATURE ANALYSIS (Logistic Regression)")
    print("=" * 80)

    # Train one pipeline per condition
    pipes: dict[str, Pipeline] = {}
    for cond in TEXT_CONDITIONS:
        col = TEXT_COLUMNS[cond]
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2), min_df=5, max_df=0.9,
                stop_words="english",
            )),
            ("clf", LogisticRegression(
                max_iter=1000, random_state=RANDOM_STATE
            )),
        ])
        pipe.fit(train_df[col], train_df["label"])
        pipes[cond] = pipe

    # Compare each masked condition against original
    analyses = {}
    for cond in ["structural", "random", "topical"]:
        col = TEXT_COLUMNS[cond]
        analyses[cond] = create_ablation_analysis_csv(
            test_df,
            original_pipe=pipes["original"],
            masked_pipe=pipes[cond],
            masked_col=col,
            output_name=cond,
        )

    # ── Top-20 features for original vs structural ──
    for cond in ["original", "structural"]:
        vec = pipes[cond].named_steps["tfidf"]
        clf = pipes[cond].named_steps["clf"]
        feats = vec.get_feature_names_out()
        coefs = clf.coef_[0]

        top_female = np.argsort(coefs)[-20:][::-1]
        top_male = np.argsort(coefs)[:20]

        female_df = pd.DataFrame(
            [(feats[i], coefs[i]) for i in top_female],
            columns=["feature", "weight"],
        )
        male_df = pd.DataFrame(
            [(feats[i], coefs[i]) for i in top_male],
            columns=["feature", "weight"],
        )

        female_df.to_csv(
            os.path.join(RESULTS_DIR, f"top_features_{cond}_female.csv"),
            index=False,
        )
        male_df.to_csv(
            os.path.join(RESULTS_DIR, f"top_features_{cond}_male.csv"),
            index=False,
        )
        print(f"\n  Top-20 features saved for {cond} (male & female).")

    return analyses
