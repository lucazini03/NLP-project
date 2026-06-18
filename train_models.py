import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
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


def run_transformer_experiments(train_df, test_df, text_column="text"):
    """
    Fine-tune distilbert-base-uncased using Hugging Face transformers and datasets.
    """
    model_name = "distilbert-base-uncased"
    print(f"\n" + "=" * 80)
    print(f"TRANSFORMER EXPERIMENTS: {model_name} on {text_column}")
    print("=" * 80)
    
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    unique_labels = train_df["label"].unique()
    label2id = {str(label): i for i, label in enumerate(sorted(unique_labels))}
    id2label = {i: str(label) for label, i in label2id.items()}
    num_labels = len(unique_labels)
    
    train_dataset = Dataset.from_pandas(train_df[[text_column, "label"]])
    test_dataset = Dataset.from_pandas(test_df[[text_column, "label"]])
    
    def tokenize_function(examples):
        tokens = tokenizer(examples[text_column], padding="max_length", truncation=True, max_length=128)
        tokens["labels"] = [label2id[str(label)] for label in examples["label"]]
        return tokens
        
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, label2id=label2id, id2label=id2label
    )
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="macro")
        return {"accuracy": acc, "macro_f1": f1}
        
    training_args = TrainingArguments(
        output_dir=f"./results_transformer_{text_column}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        logging_steps=50,
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    
    print(f"\nTraining model on {text_column} for 3 epochs...")
    trainer.train()
    
    print(f"\nEvaluating model on {text_column}...")
    eval_results = trainer.evaluate()
    
    acc = eval_results["eval_accuracy"]
    f1 = eval_results["eval_macro_f1"]
    
    print(f"Results for {text_column} -> Accuracy: {acc:.4f}, Macro-F1: {f1:.4f}")
    
    settings_map = {
        "text": "original",
        "text_leakage_masked": "leakage_masked",
        "text_noun_masked": "noun_masked"
    }
    setting_name = settings_map.get(text_column, text_column)
    
    return {
        "model": "distilbert-base-uncased",
        "setting": setting_name,
        "features": "transformer",
        "accuracy": acc,
        "macro_f1": f1
    }