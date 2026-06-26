"""
train_models.py – Cross-validated training with N×M evaluation matrix.

Implements:
  • StratifiedGroupKFold (5-fold, author-disjoint, seeded).
  • Majority-class baseline per fold (DummyClassifier).
  • 4×4 Train×Test matrix for NB, LR, SVM (+ ModernBERT via HF Trainer).
  • Standard metrics: Mean ± Std Accuracy & Macro-F1, 95% CI (t-dist, df=4).
  • Counterfactual Flip Rate for diagnostic evaluations.
  • Corrected resampled paired t-test (Nadeau & Bengio, 2003) + Cohen's d.
  • Persistence: models & results saved to Drive-backed directories.
"""
import os
import json
import logging
import pickle
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score

from config import (
    RANDOM_STATE, N_FOLDS, TEXT_CONDITIONS, TEXT_COLUMNS,
    MODELS_DIR, RESULTS_DIR, get_rng,
)

logger = logging.getLogger(__name__)

# ─────────────────── Helpers ──────────────────────────────────

def _ensure_dirs():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


def _make_tfidf_pipeline(clf) -> Pipeline:
    """Standard TF-IDF pipeline used by all classical models."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.9,
            stop_words="english",
        )),
        ("clf", clf),
    ])


def _get_classical_models() -> dict[str, Any]:
    """Return fresh instances of classical models (seeded where applicable)."""
    return {
        "MultinomialNB": MultinomialNB(),
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "LinearSVM": LinearSVC(
            random_state=RANDOM_STATE, max_iter=5000
        ),
    }


# ─────────── Corrected resampled paired t-test ────────────────
# Nadeau & Bengio (2003), "Inference for the Generalization Error"

def corrected_paired_ttest(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_train: int,
    n_test: int,
) -> tuple[float, float]:
    """Corrected resampled paired t-test (Nadeau & Bengio, 2003).

    Parameters
    ----------
    scores_a, scores_b : array-like, shape (k,)
        Per-fold metric values for the two conditions.
    n_train, n_test : int
        Number of training / test samples per fold (approximate; we use
        the mean across folds for the ratio).

    Returns
    -------
    t_stat, p_value
    """
    diffs = np.array(scores_a) - np.array(scores_b)
    k = len(diffs)
    d_bar = diffs.mean()
    s2 = diffs.var(ddof=1)

    # Correction factor: accounts for non-independence of fold errors
    correction = (1 / k) + (n_test / n_train)
    t_stat = d_bar / np.sqrt(correction * s2) if s2 > 0 else 0.0
    # Degrees of freedom = k - 1
    p_value = 2 * sp_stats.t.sf(abs(t_stat), df=k - 1)
    return float(t_stat), float(p_value)


def cohens_d(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """Compute Cohen's d for paired samples."""
    diffs = np.array(scores_a) - np.array(scores_b)
    return float(diffs.mean() / diffs.std(ddof=1)) if diffs.std(ddof=1) > 0 else 0.0


def confidence_interval_95(scores: np.ndarray) -> tuple[float, float]:
    """95% CI using t-distribution with k-1 degrees of freedom."""
    k = len(scores)
    mean = scores.mean()
    se = scores.std(ddof=1) / np.sqrt(k)
    t_crit = sp_stats.t.ppf(0.975, df=k - 1)
    return (float(mean - t_crit * se), float(mean + t_crit * se))


# ───────────── Classical model training per fold ──────────────

def _train_and_evaluate_fold(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    fold_i: int,
    save_models: bool = True,
) -> dict:
    """Train all classical models on all 4 text conditions for one fold.

    Returns a nested dict:
        results[model_name][train_cond][test_cond] = {
            "accuracy": float,
            "macro_f1": float,
            "predictions": np.ndarray,
        }
    Also includes "baseline" with majority-class results.
    """
    _ensure_dirs()

    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]
    y_train = train_df["label"].values
    y_test = test_df["label"].values

    results: dict = {}

    # ── Majority-class baseline ──
    dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    dummy.fit(np.zeros((len(y_train), 1)), y_train)
    dummy_preds = dummy.predict(np.zeros((len(y_test), 1)))
    baseline_acc = accuracy_score(y_test, dummy_preds)
    baseline_f1 = f1_score(y_test, dummy_preds, average="macro")
    results["Baseline"] = {
        "accuracy": baseline_acc,
        "macro_f1": baseline_f1,
    }
    logger.info("Fold %d baseline: acc=%.4f, f1=%.4f", fold_i, baseline_acc, baseline_f1)

    # ── Classical models: 4×4 matrix ──
    models = _get_classical_models()

    for model_name, clf_instance in models.items():
        results[model_name] = {}

        for train_cond in TEXT_CONDITIONS:
            train_col = TEXT_COLUMNS[train_cond]
            X_train = train_df[train_col].values

            # Build & fit pipeline
            pipe = _make_tfidf_pipeline(clf_instance.__class__(
                **clf_instance.get_params()
            ))
            pipe.fit(X_train, y_train)

            # Optionally save the trained pipeline
            if save_models:
                model_path = os.path.join(
                    MODELS_DIR,
                    f"fold{fold_i}_{model_name}_{train_cond}.pkl",
                )
                with open(model_path, "wb") as f:
                    pickle.dump(pipe, f)

            results[model_name][train_cond] = {}

            for test_cond in TEXT_CONDITIONS:
                test_col = TEXT_COLUMNS[test_cond]
                X_test = test_df[test_col].values
                preds = pipe.predict(X_test)
                acc = accuracy_score(y_test, preds)
                f1 = f1_score(y_test, preds, average="macro")

                results[model_name][train_cond][test_cond] = {
                    "accuracy": acc,
                    "macro_f1": f1,
                    "predictions": preds,
                }

        print(f"  Fold {fold_i} | {model_name} done.")

    return results


# ────────────── Aggregate results across folds ────────────────

def run_cross_validation(
    df: pd.DataFrame,
    folds: list[dict],
    save_models: bool = True,
) -> dict:
    """Run the full N×M cross-validation pipeline for classical models.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with all text columns (text, text_structural,
        text_random, text_topical) and 'label', 'author_id'.
    folds : list[dict]
        Output of prepare_data.generate_folds().
    save_models : bool
        Whether to pickle models to MODELS_DIR.

    Returns
    -------
    A dict with:
        "fold_results": list of per-fold result dicts,
        "summary": pd.DataFrame with Mean±Std, 95% CI for every cell,
        "flip_rates": pd.DataFrame,
        "significance_tests": dict,
    """
    _ensure_dirs()
    all_fold_results = []

    print("\n" + "=" * 80)
    print("CROSS-VALIDATED TRAINING: Classical Models (4×4 matrix)")
    print("=" * 80)

    for fold_info in folds:
        fold_i = fold_info["fold"]
        print(f"\n─── Fold {fold_i} ───")
        res = _train_and_evaluate_fold(
            df,
            fold_info["train_idx"],
            fold_info["test_idx"],
            fold_i,
            save_models=save_models,
        )
        all_fold_results.append(res)

    # ── Build summary tables ──
    model_names = [m for m in all_fold_results[0] if m != "Baseline"]
    summary_rows = []

    for model_name in model_names:
        for train_cond in TEXT_CONDITIONS:
            for test_cond in TEXT_CONDITIONS:
                accs = np.array([
                    fr[model_name][train_cond][test_cond]["accuracy"]
                    for fr in all_fold_results
                ])
                f1s = np.array([
                    fr[model_name][train_cond][test_cond]["macro_f1"]
                    for fr in all_fold_results
                ])
                ci_acc = confidence_interval_95(accs)
                ci_f1 = confidence_interval_95(f1s)

                summary_rows.append({
                    "model": model_name,
                    "train_condition": train_cond,
                    "test_condition": test_cond,
                    "mean_accuracy": accs.mean(),
                    "std_accuracy": accs.std(ddof=1),
                    "ci95_acc_lo": ci_acc[0],
                    "ci95_acc_hi": ci_acc[1],
                    "mean_macro_f1": f1s.mean(),
                    "std_macro_f1": f1s.std(ddof=1),
                    "ci95_f1_lo": ci_f1[0],
                    "ci95_f1_hi": ci_f1[1],
                })

    # Baseline row
    baseline_accs = np.array([fr["Baseline"]["accuracy"] for fr in all_fold_results])
    baseline_f1s = np.array([fr["Baseline"]["macro_f1"] for fr in all_fold_results])
    ci_b_acc = confidence_interval_95(baseline_accs)
    ci_b_f1 = confidence_interval_95(baseline_f1s)
    summary_rows.append({
        "model": "Baseline (majority)",
        "train_condition": "N/A",
        "test_condition": "N/A",
        "mean_accuracy": baseline_accs.mean(),
        "std_accuracy": baseline_accs.std(ddof=1),
        "ci95_acc_lo": ci_b_acc[0],
        "ci95_acc_hi": ci_b_acc[1],
        "mean_macro_f1": baseline_f1s.mean(),
        "std_macro_f1": baseline_f1s.std(ddof=1),
        "ci95_f1_lo": ci_b_f1[0],
        "ci95_f1_hi": ci_b_f1[1],
    })

    summary_df = pd.DataFrame(summary_rows)

    # ── Flip rates (diagnostic: Train Original → Test Masked) ──
    flip_rows = []
    for model_name in model_names:
        for test_cond in ["structural", "random", "topical"]:
            flip_rates = []
            for fr in all_fold_results:
                preds_orig = fr[model_name]["original"]["original"]["predictions"]
                preds_masked = fr[model_name]["original"][test_cond]["predictions"]
                flip_rate = (preds_orig != preds_masked).mean()
                flip_rates.append(flip_rate)
            flip_arr = np.array(flip_rates)
            flip_rows.append({
                "model": model_name,
                "test_condition": test_cond,
                "mean_flip_rate": flip_arr.mean(),
                "std_flip_rate": flip_arr.std(ddof=1),
            })

    flip_df = pd.DataFrame(flip_rows)

    # ── Significance tests (corrected paired t-test) ──
    sig_results = {}
    for model_name in model_names:
        sig_results[model_name] = {}

        # Compute average fold sizes for the correction factor
        avg_n_train = np.mean([len(f["train_idx"]) for f in folds])
        avg_n_test = np.mean([len(f["test_idx"]) for f in folds])

        for cmp_cond in ["structural", "random"]:
            accs_orig = np.array([
                fr[model_name]["original"]["original"]["accuracy"]
                for fr in all_fold_results
            ])
            accs_masked = np.array([
                fr[model_name]["original"][cmp_cond]["accuracy"]
                for fr in all_fold_results
            ])

            t_stat, p_val = corrected_paired_ttest(
                accs_orig, accs_masked,
                n_train=int(avg_n_train),
                n_test=int(avg_n_test),
            )
            d = cohens_d(accs_orig, accs_masked)

            sig_results[model_name][f"original_vs_{cmp_cond}"] = {
                "t_statistic": t_stat,
                "p_value": p_val,
                "cohens_d": d,
                "mean_diff": float(accs_orig.mean() - accs_masked.mean()),
            }

    # ── Save everything ──
    summary_df.to_csv(
        os.path.join(RESULTS_DIR, "cv_summary.csv"), index=False
    )
    flip_df.to_csv(
        os.path.join(RESULTS_DIR, "flip_rates.csv"), index=False
    )
    with open(os.path.join(RESULTS_DIR, "significance_tests.json"), "w") as f:
        json.dump(sig_results, f, indent=2)

    # ── Pretty-print ──
    _print_summary(summary_df, flip_df, sig_results, baseline_accs, baseline_f1s)

    return {
        "fold_results": all_fold_results,
        "summary": summary_df,
        "flip_rates": flip_df,
        "significance_tests": sig_results,
    }


def _print_summary(summary_df, flip_df, sig_results, baseline_accs, baseline_f1s):
    """Pretty-print results to stdout."""
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    print(f"\nBaseline (majority): "
          f"Acc = {baseline_accs.mean():.4f} ± {baseline_accs.std(ddof=1):.4f}, "
          f"F1 = {baseline_f1s.mean():.4f} ± {baseline_f1s.std(ddof=1):.4f}")

    # Print 4×4 matrices per model
    non_baseline = summary_df[summary_df["model"] != "Baseline (majority)"]
    for model_name in non_baseline["model"].unique():
        model_df = non_baseline[non_baseline["model"] == model_name]
        print(f"\n{'─' * 60}")
        print(f"Model: {model_name}")
        print(f"{'─' * 60}")

        # Accuracy matrix
        print("\n  Accuracy (Mean ± Std):")
        for train_c in TEXT_CONDITIONS:
            row_data = []
            for test_c in TEXT_CONDITIONS:
                cell = model_df[
                    (model_df["train_condition"] == train_c) &
                    (model_df["test_condition"] == test_c)
                ].iloc[0]
                row_data.append(
                    f"{cell['mean_accuracy']:.4f}±{cell['std_accuracy']:.4f}"
                )
            print(f"    Train {train_c:12s} | " + " | ".join(row_data))

        # F1 matrix
        print("\n  Macro-F1 (Mean ± Std):")
        for train_c in TEXT_CONDITIONS:
            row_data = []
            for test_c in TEXT_CONDITIONS:
                cell = model_df[
                    (model_df["train_condition"] == train_c) &
                    (model_df["test_condition"] == test_c)
                ].iloc[0]
                row_data.append(
                    f"{cell['mean_macro_f1']:.4f}±{cell['std_macro_f1']:.4f}"
                )
            print(f"    Train {train_c:12s} | " + " | ".join(row_data))

    # Flip rates
    print(f"\n{'─' * 60}")
    print("Flip Rates (Train Original → Test Masked):")
    print(f"{'─' * 60}")
    for _, row in flip_df.iterrows():
        print(f"  {row['model']:20s} → {row['test_condition']:12s}: "
              f"{row['mean_flip_rate']:.4f} ± {row['std_flip_rate']:.4f}")

    # Significance
    print(f"\n{'─' * 60}")
    print("Corrected Paired t-test (Nadeau & Bengio):")
    print(f"{'─' * 60}")
    for model_name, tests in sig_results.items():
        for test_name, vals in tests.items():
            print(f"  {model_name} | {test_name}: "
                  f"t={vals['t_statistic']:.4f}, "
                  f"p={vals['p_value']:.4f}, "
                  f"d={vals['cohens_d']:.4f}, "
                  f"Δ={vals['mean_diff']:.4f}")


# ═══════════════════ ModernBERT / Transformer ═════════════════

def run_transformer_fold(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    fold_i: int,
    model_name: str = "answerdotai/ModernBERT-base",
    max_length: int = 256,
    epochs: int = 2,
    batch_size: int = 16,
    lr: float = 2e-5,
    early_stopping_patience: int = 1,
) -> dict:
    """Train & evaluate a transformer on all 4×4 conditions for one fold.

    Returns the same nested structure as _train_and_evaluate_fold but for
    the transformer model.
    """
    import torch
    from datasets import Dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        EarlyStoppingCallback,
    )

    _ensure_dirs()

    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]
    y_test = test_df["label"].values

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Transformer fold {fold_i} | device={device} | model={model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    unique_labels = sorted(df["label"].unique())
    label2id = {str(l): i for i, l in enumerate(unique_labels)}
    id2label = {i: str(l) for l, i in label2id.items()}
    num_labels = len(unique_labels)

    def _compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro"),
        }

    results: dict = {}

    for train_cond in TEXT_CONDITIONS:
        train_col = TEXT_COLUMNS[train_cond]

        # Prepare HF datasets
        _train_ds = Dataset.from_dict({
            "text": train_df[train_col].tolist(),
            "label": [label2id[str(l)] for l in train_df["label"]],
        })

        def _tok(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length,
            )

        _train_ds = _train_ds.map(_tok, batched=True)
        _train_ds = _train_ds.rename_column("label", "labels")
        _train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

        # Split 90/10 for early stopping validation
        split = _train_ds.train_test_split(test_size=0.1, seed=RANDOM_STATE)
        _actual_train_ds = split["train"]
        _val_ds = split["test"]

        # Fresh model per train condition
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
        )

        output_dir = os.path.join(
            MODELS_DIR,
            f"transformer_fold{fold_i}_{train_cond}",
        )

        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            seed=RANDOM_STATE,
            logging_steps=100,
            report_to="none",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=2,
            fp16=torch.cuda.is_available(),
        )

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=_actual_train_ds,
            eval_dataset=_val_ds,
            data_collator=data_collator,
            compute_metrics=_compute_metrics,
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
            )],
        )

        print(f"    Training on {train_cond}...")
        trainer.train()

        results[train_cond] = {}

        for test_cond in TEXT_CONDITIONS:
            test_col = TEXT_COLUMNS[test_cond]
            _test_ds = Dataset.from_dict({
                "text": test_df[test_col].tolist(),
                "label": [label2id[str(l)] for l in test_df["label"]],
            })
            _test_ds = _test_ds.map(_tok, batched=True)
            _test_ds = _test_ds.rename_column("label", "labels")
            _test_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

            eval_out = trainer.predict(_test_ds)
            preds = np.argmax(eval_out.predictions, axis=-1)
            acc = accuracy_score(y_test, preds)
            macro_f1 = f1_score(y_test, preds, average="macro")

            results[train_cond][test_cond] = {
                "accuracy": acc,
                "macro_f1": macro_f1,
                "predictions": preds,
            }

        print(f"    Train {train_cond} – all test conditions evaluated.")

    return results


def run_transformer_cv(
    df: pd.DataFrame,
    folds: list[dict],
    model_name: str = "answerdotai/ModernBERT-base",
    **kwargs,
) -> dict:
    """Run full 4×4 CV for a transformer model across all folds.

    Returns same structure as run_cross_validation().
    """
    _ensure_dirs()
    all_fold_results = []

    print("\n" + "=" * 80)
    print(f"CROSS-VALIDATED TRAINING: {model_name} (4×4 matrix)")
    print("=" * 80)

    for fold_info in folds:
        fold_i = fold_info["fold"]
        print(f"\n─── Fold {fold_i} ───")
        res = run_transformer_fold(
            df,
            fold_info["train_idx"],
            fold_info["test_idx"],
            fold_i,
            model_name=model_name,
            **kwargs,
        )
        all_fold_results.append(res)

    # ── Build summary tables (same logic as classical) ──
    summary_rows = []
    short_name = model_name.split("/")[-1]

    for train_cond in TEXT_CONDITIONS:
        for test_cond in TEXT_CONDITIONS:
            accs = np.array([
                fr[train_cond][test_cond]["accuracy"]
                for fr in all_fold_results
            ])
            f1s = np.array([
                fr[train_cond][test_cond]["macro_f1"]
                for fr in all_fold_results
            ])
            ci_acc = confidence_interval_95(accs)
            ci_f1 = confidence_interval_95(f1s)

            summary_rows.append({
                "model": short_name,
                "train_condition": train_cond,
                "test_condition": test_cond,
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

    # Flip rates
    flip_rows = []
    for test_cond in ["structural", "random", "topical"]:
        flip_rates = []
        for fr in all_fold_results:
            preds_orig = fr["original"]["original"]["predictions"]
            preds_masked = fr["original"][test_cond]["predictions"]
            flip_rate = (preds_orig != preds_masked).mean()
            flip_rates.append(flip_rate)
        flip_arr = np.array(flip_rates)
        flip_rows.append({
            "model": short_name,
            "test_condition": test_cond,
            "mean_flip_rate": flip_arr.mean(),
            "std_flip_rate": flip_arr.std(ddof=1),
        })
    flip_df = pd.DataFrame(flip_rows)

    # Significance
    sig_results = {}
    avg_n_train = np.mean([len(f["train_idx"]) for f in folds])
    avg_n_test = np.mean([len(f["test_idx"]) for f in folds])

    for cmp_cond in ["structural", "random"]:
        accs_orig = np.array([
            fr["original"]["original"]["accuracy"]
            for fr in all_fold_results
        ])
        accs_masked = np.array([
            fr["original"][cmp_cond]["accuracy"]
            for fr in all_fold_results
        ])
        t_stat, p_val = corrected_paired_ttest(
            accs_orig, accs_masked,
            n_train=int(avg_n_train),
            n_test=int(avg_n_test),
        )
        d = cohens_d(accs_orig, accs_masked)
        sig_results[f"original_vs_{cmp_cond}"] = {
            "t_statistic": t_stat,
            "p_value": p_val,
            "cohens_d": d,
            "mean_diff": float(accs_orig.mean() - accs_masked.mean()),
        }

    # Save
    summary_df.to_csv(
        os.path.join(RESULTS_DIR, f"cv_summary_{short_name}.csv"), index=False
    )
    flip_df.to_csv(
        os.path.join(RESULTS_DIR, f"flip_rates_{short_name}.csv"), index=False
    )
    with open(os.path.join(RESULTS_DIR, f"significance_{short_name}.json"), "w") as f:
        json.dump(sig_results, f, indent=2)

    _print_summary(summary_df, flip_df, {short_name: sig_results},
                   np.zeros(N_FOLDS), np.zeros(N_FOLDS))

    return {
        "fold_results": all_fold_results,
        "summary": summary_df,
        "flip_rates": flip_df,
        "significance_tests": sig_results,
    }