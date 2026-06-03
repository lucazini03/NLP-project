import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def get_top_features_for_text(text, vectorizer, clf, top_n=5):
    """
    Return the most influential features for a single text prediction.
    """
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


def create_masked_analysis_csv(
    test_df,
    original_pipe,
    masked_pipe,
    masked_col,
    output_name,
    sample_size=200,
    n_examples=5
):
    """
    Compare original and masked models and save a qualitative analysis CSV.
    """
    print(f"CREATING {output_name.upper()} ANALYSIS CSV")


    if len(test_df) > sample_size:
        test_sample = test_df.sample(sample_size, random_state=42).reset_index(drop=True)
    else:
        test_sample = test_df.reset_index(drop=True)

    orig_preds = original_pipe.predict(test_sample["text"])
    mask_preds = masked_pipe.predict(test_sample[masked_col])

    orig_proba = original_pipe.predict_proba(test_sample["text"])
    mask_proba = masked_pipe.predict_proba(test_sample[masked_col])

    vec_orig = original_pipe.named_steps["tfidf"]
    clf_orig = original_pipe.named_steps["clf"]
    vec_mask = masked_pipe.named_steps["tfidf"]
    clf_mask = masked_pipe.named_steps["clf"]

    rows = []

    for i, row in test_sample.iterrows():
        orig_feats = get_top_features_for_text(
            row["text"], vec_orig, clf_orig
        )
        mask_feats = get_top_features_for_text(
            row[masked_col], vec_mask, clf_mask
        )

        rows.append({
            "original_text": row["text"][:200],
            "masked_text": row[masked_col][:200],
            "true_label": "Female" if row["label"] == 1 else "Male",

            "original_prediction": "Female" if orig_preds[i] == 1 else "Male",
            "original_correct": orig_preds[i] == row["label"],
            "original_confidence": f"{orig_proba[i][orig_preds[i]]:.3f}",
            "original_top_features": ", ".join(
                f"{f} ({w:+.2f})" for f, w in orig_feats
            ),

            "masked_prediction": "Female" if mask_preds[i] == 1 else "Male",
            "masked_correct": mask_preds[i] == row["label"],
            "masked_confidence": f"{mask_proba[i][mask_preds[i]]:.3f}",
            "masked_top_features": ", ".join(
                f"{f} ({w:+.2f})" for f, w in mask_feats
            ),

            "prediction_changed": orig_preds[i] != mask_preds[i]
        })

    df = pd.DataFrame(rows)
    df.to_csv(f"results/{output_name}_analysis.csv", index=False)


    print(f"\n[{output_name.upper()} SUMMARY]")
    print(f"Original accuracy: {df['original_correct'].mean():.2%}")
    print(f"Masked accuracy:   {df['masked_correct'].mean():.2%}")
    print(f"Predictions changed: {df['prediction_changed'].mean():.1%}")


    flipped = df[df["prediction_changed"]].head(n_examples)


    print(f"INTERESTING CASES – {output_name.upper()} (Prediction Changed)")


    if len(flipped) == 0:
        print("No prediction changes found.")
    else:
        for i, row in flipped.iterrows():
            print(f"\nCase {i}")
            print(f"  True label: {row['true_label']}")
            print(
                f"  Prediction: {row['original_prediction']} → "
                f"{row['masked_prediction']}"
            )
            print(f"  Original text: {row['original_text'][:120]}...")
            print(f"  Original features: {row['original_top_features']}")
            print(f"  Masked features:   {row['masked_top_features']}")

    return df




def analyze_features(train_df, test_df):
    """
    Train original and masked models and run qualitative feature analysis.
    """
    print("QUALITATIVE ANALYSIS: Running models on samples")

    original_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=5, stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    original_pipe.fit(train_df["text"], train_df["label"])

    leakage_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=5, stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    leakage_pipe.fit(train_df["text_leakage_masked"], train_df["label"])

    noun_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=5, stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    noun_pipe.fit(train_df["text_noun_masked"], train_df["label"])

    create_masked_analysis_csv(
        test_df,
        original_pipe,
        leakage_pipe,
        masked_col="text_leakage_masked",
        output_name="leakage"
    )

    create_masked_analysis_csv(
        test_df,
        original_pipe,
        noun_pipe,
        masked_col="text_noun_masked",
        output_name="noun"
    )

    masked_pipe = leakage_pipe
    masked_text_col = "text_leakage_masked"

    vec_orig = original_pipe.named_steps["tfidf"]
    clf_orig = original_pipe.named_steps["clf"]
    vec_mask = masked_pipe.named_steps["tfidf"]
    clf_mask = masked_pipe.named_steps["clf"]

    feats_orig = vec_orig.get_feature_names_out()
    feats_mask = vec_mask.get_feature_names_out()

    coefs_orig = clf_orig.coef_[0]
    coefs_mask = clf_mask.coef_[0]

    top_orig = np.argsort(coefs_orig)[-20:][::-1]
    top_mask = np.argsort(coefs_mask)[-20:][::-1]

    original_female = [(feats_orig[i], coefs_orig[i]) for i in top_orig]
    masked_female = [(feats_mask[i], coefs_mask[i]) for i in top_mask]

    pd.DataFrame(original_female, columns=["feature", "weight"]).to_csv(
        "results/original_features.csv", index=False
    )
    pd.DataFrame(masked_female, columns=["feature", "weight"]).to_csv(
        "results/masked_features.csv", index=False
    )


    orig_preds = original_pipe.predict(test_df["text"])
    mask_preds = masked_pipe.predict(test_df[masked_text_col])
