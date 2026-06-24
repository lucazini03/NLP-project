import spacy
from tqdm import tqdm
import pandas as pd

def extract_spacy_features(texts, batch_size=50, chunk_size=2000, model_name="en_core_web_sm"):
    """
    Extract simple spaCy-based stylometric features for a list of texts.
    Returns a pandas DataFrame (one row per text) with numeric features.
    """
    nlp = spacy.load(model_name, disable=["ner"])
    pos_tags = ["NOUN","PROPN","VERB","ADJ","ADV","PRON","ADP","DET","NUM","PUNCT","SYM","X"]

    rows = []
    for start in range(0, len(texts), chunk_size):
        chunk = texts[start:start + chunk_size]
        for doc in nlp.pipe(chunk, batch_size=batch_size):
            token_count = len(doc)
            sent_count = sum(1 for _ in doc.sents) or 1
            avg_sent_len = token_count / sent_count if sent_count else token_count

            pos_counts = {f"pos_{p.lower()}": 0 for p in pos_tags}
            stop_words = 0
            uppercase = 0
            num_exclam = 0
            num_question = 0
            total_token_len = 0

            for token in doc:
                tp = token.pos_
                if tp in pos_counts:
                    pos_counts[f"pos_{tp.lower()}"] += 1
                if token.is_stop:
                    stop_words += 1
                if token.text.isupper() and token.text.isalpha():
                    uppercase += 1
                if token.text == "!":
                    num_exclam += 1
                elif token.text == "?":
                    num_question += 1
                total_token_len += len(token.text)

            avg_token_len = total_token_len / token_count if token_count else 0

            row = {
                "token_count": token_count,
                "sent_count": sent_count,
                "avg_sent_len": avg_sent_len,
                "avg_token_len": avg_token_len,
                "stop_words": stop_words,
                "uppercase_tokens": uppercase,
                "num_exclam": num_exclam,
                "num_question": num_question,
            }
            row.update(pos_counts)
            rows.append(row)

    df = pd.DataFrame(rows).fillna(0)
    return df