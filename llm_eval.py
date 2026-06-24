"""
llm_eval.py – Zero-shot LLM evaluation on the 4 counterfactual text conditions.

Supports:
  • Google Generative AI (Gemini 1.5 Flash)  – default
  • OpenAI (GPT-4o-mini)                     – optional

Design:
  • No training – zero-shot classification via a fixed prompt.
  • Stratified subsample of N=200 docs per fold (seeded), same indices
    across all 4 text versions for strict paired comparison.
  • Disk-backed JSON cache: (doc_id, text_version, model, prompt) → response.
  • temperature=0, parse "Male" / "Female" from raw response.
  • Reports Mean/Std Accuracy, Macro-F1, and Flip Rate across folds.
"""
import os
import re
import json
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from config import (
    RANDOM_STATE, LLM_SUBSAMPLE_N, LLM_TEMPERATURE, LLM_PROMPT,
    TEXT_CONDITIONS, TEXT_COLUMNS, LLM_CACHE_DIR, RESULTS_DIR, N_FOLDS,
    get_rng,
)
from train_models import confidence_interval_95

logger = logging.getLogger(__name__)

# ─────────────────── Cache ────────────────────────────────────

def _cache_path(provider: str) -> Path:
    os.makedirs(LLM_CACHE_DIR, exist_ok=True)
    return Path(LLM_CACHE_DIR) / f"llm_cache_{provider}.json"


def _load_cache(provider: str) -> dict:
    path = _cache_path(provider)
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict, provider: str):
    path = _cache_path(provider)
    with open(path, "w") as f:
        json.dump(cache, f, indent=2)


def _cache_key(doc_id: int, text_version: str, model: str) -> str:
    return f"{doc_id}|{text_version}|{model}"


# ─────────────────── Response parsing ─────────────────────────

def _parse_gender(raw_response: str) -> int | None:
    """Parse 'Male' → 0, 'Female' → 1, else None."""
    text = raw_response.strip().lower()
    # Try to extract just the first word
    match = re.search(r"\b(male|female)\b", text)
    if match:
        return 0 if match.group(1) == "male" else 1
    return None


# ─────────────────── Provider abstraction ─────────────────────

class LLMProvider:
    """Base class. Subclass and implement `_call_api`."""

    provider_name: str = "base"
    model_id: str = ""

    def _call_api(self, prompt_text: str) -> str:
        raise NotImplementedError

    def classify(self, text: str) -> str:
        """Fill the prompt template and call the API."""
        full_prompt = LLM_PROMPT.format(text=text)
        return self._call_api(full_prompt)


class GeminiProvider(LLMProvider):
    """Google Generative AI (Gemini)."""

    provider_name = "gemini"

    def __init__(self, model_id: str = "gemini-1.5-flash"):
        import google.generativeai as genai  # type: ignore
        self.model_id = model_id
        self._model = genai.GenerativeModel(model_id)
        self._genai = genai

    def _call_api(self, prompt_text: str) -> str:
        response = self._model.generate_content(
            prompt_text,
            generation_config=self._genai.types.GenerationConfig(
                temperature=LLM_TEMPERATURE,
                max_output_tokens=10,
            ),
        )
        return response.text.strip()


class OpenAIProvider(LLMProvider):
    """OpenAI API (e.g. GPT-4o-mini)."""

    provider_name = "openai"

    def __init__(self, model_id: str = "gpt-4o-mini"):
        from openai import OpenAI  # type: ignore
        self.model_id = model_id
        self._client = OpenAI()

    def _call_api(self, prompt_text: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt_text}],
            temperature=LLM_TEMPERATURE,
            max_tokens=10,
        )
        return response.choices[0].message.content.strip()


# ─────────────── Subsample selection ──────────────────────────

def _stratified_subsample(
    y: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return indices for a stratified random sample of size n."""
    classes = np.unique(y)
    selected = []
    per_class = n // len(classes)
    remainder = n - per_class * len(classes)

    for i, c in enumerate(classes):
        class_idx = np.where(y == c)[0]
        k = per_class + (1 if i < remainder else 0)
        k = min(k, len(class_idx))
        chosen = rng.choice(class_idx, size=k, replace=False)
        selected.extend(chosen.tolist())

    return np.array(sorted(selected))


# ─────────────── Main evaluation loop ─────────────────────────

def evaluate_llm(
    df: pd.DataFrame,
    folds: list[dict],
    provider: LLMProvider | None = None,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> dict:
    """Run zero-shot LLM evaluation across all folds and text conditions.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with all text columns and 'label'.
    folds : list[dict]
        Output of prepare_data.generate_folds().
    provider : LLMProvider
        An instantiated LLM provider. Defaults to GeminiProvider().

    Returns
    -------
    dict with "summary" (pd.DataFrame), "flip_rates" (pd.DataFrame).
    """
    if provider is None:
        provider = GeminiProvider()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    cache = _load_cache(provider.provider_name)
    rng = get_rng(RANDOM_STATE)

    # Per-fold, per-condition results
    fold_metrics: dict[str, list[dict]] = {cond: [] for cond in TEXT_CONDITIONS}
    fold_flip_rates: dict[str, list[float]] = {
        cond: [] for cond in TEXT_CONDITIONS if cond != "original"
    }

    print("\n" + "=" * 80)
    print(f"LLM EVALUATION: {provider.model_id} (zero-shot, N={LLM_SUBSAMPLE_N}/fold)")
    print("=" * 80)

    for fold_info in folds:
        fold_i = fold_info["fold"]
        test_idx = fold_info["test_idx"]
        test_df = df.iloc[test_idx].reset_index(drop=True)
        y_test = test_df["label"].values

        # Stratified subsample (same indices across conditions)
        sample_idx = _stratified_subsample(y_test, LLM_SUBSAMPLE_N, rng)
        y_sample = y_test[sample_idx]

        print(f"\n─── Fold {fold_i} ({len(sample_idx)} sampled docs) ───")

        all_preds: dict[str, np.ndarray] = {}

        for cond in TEXT_CONDITIONS:
            col = TEXT_COLUMNS[cond]
            preds = []
            texts = test_df[col].values[sample_idx]

            for local_i, (text, global_pos) in enumerate(
                zip(texts, sample_idx)
            ):
                # Use the original DataFrame index as doc_id for cache stability
                doc_id = int(test_idx[global_pos])
                key = _cache_key(doc_id, cond, provider.model_id)

                if key in cache:
                    raw = cache[key]["raw_response"]
                else:
                    # Call API with retries
                    raw = None
                    for attempt in range(max_retries):
                        try:
                            raw = provider.classify(text)
                            break
                        except Exception as e:
                            logger.warning(
                                "API error (attempt %d/%d): %s",
                                attempt + 1, max_retries, e,
                            )
                            if attempt < max_retries - 1:
                                time.sleep(retry_delay * (attempt + 1))
                    if raw is None:
                        raw = ""

                    cache[key] = {
                        "doc_id": doc_id,
                        "text_version": cond,
                        "model": provider.model_id,
                        "prompt": LLM_PROMPT.format(text=text[:200] + "..."),
                        "raw_response": raw,
                    }
                    # Persist after every request
                    _save_cache(cache, provider.provider_name)

                parsed = _parse_gender(raw)
                if parsed is None:
                    # Default to majority class when parsing fails
                    parsed = int(np.bincount(y_sample).argmax())
                    logger.warning(
                        "Could not parse response for doc %d, cond %s: '%s'. "
                        "Defaulting to majority class %d.",
                        doc_id, cond, raw, parsed,
                    )
                preds.append(parsed)

            preds_arr = np.array(preds)
            all_preds[cond] = preds_arr

            acc = accuracy_score(y_sample, preds_arr)
            f1 = f1_score(y_sample, preds_arr, average="macro")
            fold_metrics[cond].append({"accuracy": acc, "macro_f1": f1})
            print(f"  {cond:12s}  Acc={acc:.4f}  F1={f1:.4f}")

        # Flip rates
        for cond in ["structural", "random", "topical"]:
            flip = (all_preds["original"] != all_preds[cond]).mean()
            fold_flip_rates[cond].append(flip)
            print(f"  Flip rate (original→{cond}): {flip:.4f}")

    # ── Aggregate ──
    summary_rows = []
    for cond in TEXT_CONDITIONS:
        accs = np.array([m["accuracy"] for m in fold_metrics[cond]])
        f1s = np.array([m["macro_f1"] for m in fold_metrics[cond]])
        ci_acc = confidence_interval_95(accs)
        ci_f1 = confidence_interval_95(f1s)

        summary_rows.append({
            "model": provider.model_id,
            "test_condition": cond,
            "mean_accuracy": accs.mean(),
            "std_accuracy": accs.std(ddof=1),
            "ci95_acc_lo": ci_acc[0],
            "ci95_acc_hi": ci_acc[1],
            "mean_macro_f1": f1s.mean(),
            "std_macro_f1": f1s.std(ddof=1),
            "ci95_f1_lo": ci_f1[0],
            "ci95_f1_hi": ci_f1[1],
        })

    summary_df = pd.DataFrame(summary_rows)

    flip_rows = []
    for cond in ["structural", "random", "topical"]:
        arr = np.array(fold_flip_rates[cond])
        flip_rows.append({
            "model": provider.model_id,
            "test_condition": cond,
            "mean_flip_rate": arr.mean(),
            "std_flip_rate": arr.std(ddof=1),
        })
    flip_df = pd.DataFrame(flip_rows)

    # Save
    summary_df.to_csv(
        os.path.join(RESULTS_DIR, f"llm_summary_{provider.provider_name}.csv"),
        index=False,
    )
    flip_df.to_csv(
        os.path.join(RESULTS_DIR, f"llm_flip_rates_{provider.provider_name}.csv"),
        index=False,
    )

    print("\n" + "=" * 80)
    print(f"LLM SUMMARY ({provider.model_id})")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("\nFlip Rates:")
    print(flip_df.to_string(index=False))

    return {
        "summary": summary_df,
        "flip_rates": flip_df,
    }
