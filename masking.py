import spacy
from tqdm import tqdm
from config import GENDER_TERMS, LEAKAGE_TOKEN, CONTENT_TOKEN

# Load spaCy model globally
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


def mask_leakage_only(texts):
    """Remove gender-revealing leakage terms using spaCy lemmatization"""
    out = []
    for doc in tqdm(
            nlp.pipe(texts, batch_size=32),
            total=len(texts),
            desc="Leakage removal (spaCy)"
    ):
        tokens = []
        for token in doc:
            if token.lemma_.lower() in GENDER_TERMS:
                tokens.append(LEAKAGE_TOKEN)
            else:
                tokens.append(token.text)
        out.append(" ".join(tokens))
    return out


def mask_all_nouns(texts):
    """Remove all nouns and proper nouns using spaCy POS tagging"""
    out = []
    for doc in tqdm(
            nlp.pipe(texts, batch_size=32),
            total=len(texts),
            desc="Noun removal (spaCy)"
    ):
        tokens = []
        for token in doc:
            if token.pos_ in ["NOUN", "PROPN"]:
                tokens.append(CONTENT_TOKEN)
            else:
                tokens.append(token.text)
        out.append(" ".join(tokens))
    return out


def apply_masking(train_df, test_df):
    """
    Apply both leakage and noun removal using spaCy

    Creates two new columns:
    - text_leakage_masked: Gender terms removed via lemmatization
    - text_noun_masked: All nouns removed via POS tagging
    """
    # Leakage masking
    print("\n[1/4] Removing leakage terms from training set...")
    train_df.loc[:, "text_leakage_masked"] = mask_leakage_only(
        train_df["text"].tolist()
    )

    print("\n[2/4] Removing leakage terms from test set...")
    test_df.loc[:, "text_leakage_masked"] = mask_leakage_only(
        test_df["text"].tolist()
    )

    # Noun masking
    print("\n[3/4] Removing nouns from training set...")
    train_df.loc[:, "text_noun_masked"] = mask_all_nouns(
        train_df["text"].tolist()
    )

    print("\n[4/4] Removing nouns from test set...")
    test_df.loc[:, "text_noun_masked"] = mask_all_nouns(
        test_df["text"].tolist()
    )

    return train_df, test_df