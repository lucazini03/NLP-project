"""
main.py – Pipeline orchestrator for the Author Profiling experiments.

This script runs the full pipeline end-to-end:
  1. Load & clean data
  2. Generate K-fold splits
  3. Apply masking to the full dataset (once)
  4. Log masking sparsity statistics
  5. Run classical model cross-validation (4×4 matrix)
  6. (Optional) Run transformer cross-validation
  7. (Optional) Run LLM zero-shot evaluation
  8. Run qualitative analysis on last fold

All intermediate results (models, CSVs, caches) are persisted to
DRIVE_ROOT so you don't need to retrain when restarting Colab.

Usage (Colab cell or CLI):
    from main import run_pipeline
    results = run_pipeline(
        run_transformer=False,  # set True when you have GPU time
        run_llm=False,          # set True after configuring API key
    )
"""
import os
import json
import logging
import pickle

import numpy as np
import pandas as pd

from config import (
    RANDOM_STATE, RESULTS_DIR, MODELS_DIR, DRIVE_ROOT, get_rng,
)
from prepare_data import load_and_clean_data, generate_folds
from masking import apply_masking, compute_masking_stats
from train_models import run_cross_validation, run_transformer_cv
from qualitative import analyze_features

logger = logging.getLogger(__name__)


def _setup_logging():
    """Configure logging for the pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                os.path.join(DRIVE_ROOT, "pipeline.log"), mode="a"
            ),
        ],
    )


def _save_folds(folds: list[dict], path: str):
    """Serialize fold indices for reproducibility."""
    serializable = []
    for f in folds:
        serializable.append({
            "fold": f["fold"],
            "train_idx": f["train_idx"].tolist(),
            "test_idx": f["test_idx"].tolist(),
        })
    with open(path, "w") as fp:
        json.dump(serializable, fp)


def _load_folds(path: str) -> list[dict]:
    """Load previously saved fold indices."""
    with open(path, "r") as fp:
        raw = json.load(fp)
    return [
        {
            "fold": f["fold"],
            "train_idx": np.array(f["train_idx"]),
            "test_idx": np.array(f["test_idx"]),
        }
        for f in raw
    ]


def run_pipeline(
    run_transformer: bool = False,
    run_llm: bool = False,
    transformer_model: str = "answerdotai/ModernBERT-base",
    llm_provider: str = "gemini",  # "gemini" or "openai"
    force_recompute_masking: bool = False,
    force_recompute_folds: bool = False,
) -> dict:
    """Run the full experimental pipeline.

    Parameters
    ----------
    run_transformer : bool
        Whether to run the transformer CV (requires GPU, slow).
    run_llm : bool
        Whether to run the LLM evaluation (requires API key).
    transformer_model : str
        HuggingFace model name for the transformer.
    llm_provider : str
        "gemini" or "openai".
    force_recompute_masking : bool
        If True, recompute masking even if cached data exists.
    force_recompute_folds : bool
        If True, regenerate fold splits.

    Returns
    -------
    dict with all result DataFrames and dicts.
    """
    os.makedirs(DRIVE_ROOT, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    _setup_logging()

    results = {}
    rng = get_rng(RANDOM_STATE)

    # ══════════════════════════════════════════════════════════
    # STEP 1: Load & clean data
    # ══════════════════════════════════════════════════════════
    masked_data_path = os.path.join(DRIVE_ROOT, "masked_data.pkl")

    if not force_recompute_masking and os.path.exists(masked_data_path):
        print("\n✓ Loading cached masked dataset from disk...")
        with open(masked_data_path, "rb") as f:
            df = pickle.load(f)
        print(f"  Loaded {len(df)} rows with columns: {list(df.columns)}")
    else:
        print("\n" + "=" * 80)
        print("STEP 1: Loading & cleaning data")
        print("=" * 80)
        df = load_and_clean_data()

        # ══════════════════════════════════════════════════════
        # STEP 2: Apply masking (once on full dataset)
        # ══════════════════════════════════════════════════════
        print("\n" + "=" * 80)
        print("STEP 2: Generating 4 counterfactual text conditions")
        print("=" * 80)
        df = apply_masking(df, rng=rng)

        # Cache to disk
        with open(masked_data_path, "wb") as f:
            pickle.dump(df, f)
        print(f"\n✓ Masked dataset cached to {masked_data_path}")

    # ══════════════════════════════════════════════════════════
    # STEP 3: Generate folds
    # ══════════════════════════════════════════════════════════
    folds_path = os.path.join(DRIVE_ROOT, "folds.json")

    if not force_recompute_folds and os.path.exists(folds_path):
        print("\n✓ Loading cached fold splits...")
        folds = _load_folds(folds_path)
        for f in folds:
            print(f"  Fold {f['fold']}: train={len(f['train_idx'])}, "
                  f"test={len(f['test_idx'])}")
    else:
        print("\n" + "=" * 80)
        print("STEP 3: Generating StratifiedGroupKFold splits")
        print("=" * 80)
        folds = generate_folds(df)
        _save_folds(folds, folds_path)
        print(f"\n✓ Folds saved to {folds_path}")

    results["folds"] = folds

    # ══════════════════════════════════════════════════════════
    # STEP 4: Masking sparsity statistics
    # ══════════════════════════════════════════════════════════
    sparsity_path = os.path.join(RESULTS_DIR, "masking_sparsity.csv")

    if os.path.exists(sparsity_path):
        print("\n✓ Loading cached masking sparsity stats...")
        sparsity_df = pd.read_csv(sparsity_path)
    else:
        print("\n" + "=" * 80)
        print("STEP 4: Computing masking sparsity statistics")
        print("=" * 80)
        sparsity_df = compute_masking_stats(df)
        sparsity_df.to_csv(sparsity_path, index=False)

    results["sparsity"] = sparsity_df

    # ══════════════════════════════════════════════════════════
    # STEP 5: Classical model cross-validation
    # ══════════════════════════════════════════════════════════
    cv_summary_path = os.path.join(RESULTS_DIR, "cv_summary.csv")

    if os.path.exists(cv_summary_path):
        print("\n✓ Loading cached classical CV results...")
        results["classical"] = {
            "summary": pd.read_csv(cv_summary_path),
        }
        # Try to load flip rates and significance
        flip_path = os.path.join(RESULTS_DIR, "flip_rates.csv")
        sig_path = os.path.join(RESULTS_DIR, "significance_tests.json")
        if os.path.exists(flip_path):
            results["classical"]["flip_rates"] = pd.read_csv(flip_path)
        if os.path.exists(sig_path):
            with open(sig_path) as f:
                results["classical"]["significance_tests"] = json.load(f)
    else:
        print("\n" + "=" * 80)
        print("STEP 5: Classical model cross-validation (4×4 matrix)")
        print("=" * 80)
        results["classical"] = run_cross_validation(df, folds)

    # ══════════════════════════════════════════════════════════
    # STEP 6: Transformer cross-validation (optional)
    # ══════════════════════════════════════════════════════════
    if run_transformer:
        short_name = transformer_model.split("/")[-1]
        tr_summary_path = os.path.join(
            RESULTS_DIR, f"cv_summary_{short_name}.csv"
        )
        if os.path.exists(tr_summary_path):
            print(f"\n✓ Loading cached transformer results ({short_name})...")
            results["transformer"] = {
                "summary": pd.read_csv(tr_summary_path),
            }
        else:
            print(f"\n" + "=" * 80)
            print(f"STEP 6: Transformer CV ({transformer_model})")
            print("=" * 80)
            results["transformer"] = run_transformer_cv(
                df, folds, model_name=transformer_model
            )
    else:
        print("\n⏭  Skipping transformer CV (run_transformer=False)")

    # ══════════════════════════════════════════════════════════
    # STEP 7: LLM zero-shot evaluation (optional)
    # ══════════════════════════════════════════════════════════
    if run_llm:
        from llm_eval import evaluate_llm, GeminiProvider, OpenAIProvider

        if llm_provider == "openai":
            provider = OpenAIProvider()
        else:
            provider = GeminiProvider()

        llm_summary_path = os.path.join(
            RESULTS_DIR, f"llm_summary_{provider.provider_name}.csv"
        )
        if os.path.exists(llm_summary_path):
            print(f"\n✓ Loading cached LLM results ({provider.model_id})...")
            results["llm"] = {
                "summary": pd.read_csv(llm_summary_path),
            }
        else:
            print("\n" + "=" * 80)
            print(f"STEP 7: LLM evaluation ({provider.model_id})")
            print("=" * 80)
            results["llm"] = evaluate_llm(df, folds, provider=provider)
    else:
        print("\n⏭  Skipping LLM evaluation (run_llm=False)")

    # ══════════════════════════════════════════════════════════
    # STEP 8: Qualitative analysis (on last fold)
    # ══════════════════════════════════════════════════════════
    last_fold = folds[-1]
    train_df = df.iloc[last_fold["train_idx"]]
    test_df = df.iloc[last_fold["test_idx"]]

    qual_path = os.path.join(RESULTS_DIR, "structural_analysis.csv")
    if os.path.exists(qual_path):
        print("\n✓ Qualitative analysis CSVs already exist, skipping.")
    else:
        print("\n" + "=" * 80)
        print("STEP 8: Qualitative feature analysis (last fold)")
        print("=" * 80)
        results["qualitative"] = analyze_features(train_df, test_df)

    # ══════════════════════════════════════════════════════════
    # Done
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"All results saved to: {RESULTS_DIR}/")
    print(f"All models saved to:  {MODELS_DIR}/")

    return results


if __name__ == "__main__":
    run_pipeline(
        run_transformer=False,
        run_llm=False,
    )
