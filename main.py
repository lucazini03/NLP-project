from prepare_data import load_data, split_by_author, remove_near_duplicates
from masking import apply_masking
from train_models import run_experiments
from qualitative import analyze_features


def main():
    """Run the complete pipeline"""


    print("REDDIT GENDER CLASSIFICATION PIPELINE")

    # Step 1: Load and prepare data
    print("\nSTEP 1: LOAD DATA")
    df = load_data()
    train_df, test_df = split_by_author(df)
    test_df = remove_near_duplicates(train_df, test_df)

    # Step 2: Apply noun removal using spaCy
    print("\nSTEP 2: MASKING")
    train_df, test_df = apply_masking(train_df, test_df)

    # Step 3: Train and evaluate models
    print("\nSTEP 3: TRAIN MODELS")
    results_df = run_experiments(train_df, test_df)

    # Step 4: Qualitative analysis (Bilgin's part)
    print("\nSTEP 4: QUALITATIVE ANALYSIS")
    analyze_features(train_df, test_df)

    print("\nOutput files:")
    print("  - results.csv")
    print("  - original_features.csv")
    print("  - masked_features.csv")
    print("  - masking_analysis.csv")
    print("\nNote: All masking uses spaCy for POS tagging")


if __name__ == "__main__":
    main()