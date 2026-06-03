import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from spacy_features import extract_spacy_features


def run_experiments(train_df, test_df):
    """Run all experiments - same as notebook"""

    # Define models
    models = {
        "MultinomialNB": MultinomialNB(),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "LinearSVM": LinearSVC()
    }

    # Define settings
    settings = {
        "original": "text",
        "leakage_masked": "text_leakage_masked",
        "noun_masked": "text_noun_masked"
    }

    # Run experiments
    results = []
    print("\nTraining models...")

    for model_name, model in models.items():
        print(f"\n{model_name}:")
        for setting_name, column in settings.items():
            # Create pipeline
            pipe = Pipeline([
                ("tfidf", TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=5,
                    max_df=0.9,
                    stop_words="english"
                )),
                ("clf", model)
            ])

            # Train and evaluate
            pipe.fit(train_df[column], train_df["label"])
            preds = pipe.predict(test_df[column])

            acc = accuracy_score(test_df["label"], preds)
            f1 = f1_score(test_df["label"], preds, average="macro")

            results.append({
                "model": model_name,
                "setting": setting_name,
                "features": "tfidf",
                "accuracy": acc,
                "macro_f1": f1
            })

            print(f"  {setting_name:20s} Acc: {acc:.4f}, F1: {f1:.4f}")

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY (TF-IDF features)")
    print("=" * 80)
    print("\nMacro-F1:")
    print(results_df[results_df["features"] == "tfidf"].pivot(
        index="model", columns="setting", values="macro_f1"))

    print("\nAccuracy:")
    print(results_df[results_df["features"] == "tfidf"].pivot(
        index="model", columns="setting", values="accuracy"))

    # Save
    results_df.to_csv("results/results.csv", index=False)
    print("\nResults saved to results/results.csv")

    return results_df


def run_spacy_experiments(train_df, test_df):
    """
    Run experiments using spaCy-extracted linguistic features
    This demonstrates the use of spaCy for stylometric analysis
    """
    print("\n" + "=" * 80)
    print("SPACY-BASED EXPERIMENTS")
    print("=" * 80)
    print("\nExtracting spaCy features (POS tags, dependencies, syntax)...")
    
    # Extract spaCy features for original text
    print("\n[1/2] Training set...")
    train_spacy_features = extract_spacy_features(train_df["text"].tolist(), batch_size=50, chunk_size=2000)
    
    print("\n[2/2] Test set...")
    test_spacy_features = extract_spacy_features(test_df["text"].tolist(), batch_size=50, chunk_size=2000)
    
    # Save spaCy features for inspection
    train_spacy_features.to_csv("train_spacy_features.csv", index=False)
    test_spacy_features.to_csv("test_spacy_features.csv", index=False)
    print("\n✓ spaCy features saved to train_spacy_features.csv and test_spacy_features.csv")
    
    # Train models on spaCy features
    print("\n" + "=" * 80)
    print("Training models on spaCy features...")
    print("=" * 80)
    
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    spacy_results = []
    
    for model_name, model in models.items():
        print(f"\n{model_name}:")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(train_spacy_features)
        X_test_scaled = scaler.transform(test_spacy_features)
        
        # Train
        model.fit(X_train_scaled, train_df["label"])
        preds = model.predict(X_test_scaled)
        
        acc = accuracy_score(test_df["label"], preds)
        f1 = f1_score(test_df["label"], preds, average="macro")
        
        spacy_results.append({
            "model": model_name,
            "setting": "spacy_only",
            "features": "spacy_linguistic",
            "accuracy": acc,
            "macro_f1": f1
        })
        
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Macro-F1: {f1:.4f}")
    
    # Save spaCy results
    spacy_df = pd.DataFrame(spacy_results)
    spacy_df.to_csv("spacy_results.csv", index=False)
    
    print("\n" + "=" * 80)
    print("SPACY RESULTS SUMMARY")
    print("=" * 80)
    print("\nPerformance using only spaCy linguistic features:")
    print(spacy_df[["model", "accuracy", "macro_f1"]])
    print("\nInterpretation:")
    print("  - These results show performance using ONLY linguistic features")
    print("  - Features: POS tags, dependencies, punctuation, syntax")
    print("  - No lexical content (words) used - pure stylometric analysis")
    
    return spacy_df
