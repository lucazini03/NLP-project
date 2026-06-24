"""
config.py – Central configuration for the Author Profiling pipeline.

Every source of randomness in the pipeline derives from RANDOM_STATE,
either directly or via a seeded numpy.random.Generator.
"""
import numpy as np

# ─────────────────────── Reproducibility ───────────────────────
RANDOM_STATE = 42

def get_rng(seed: int | None = None) -> np.random.Generator:
    """Return a seeded numpy Generator. Uses RANDOM_STATE by default."""
    return np.random.default_rng(seed if seed is not None else RANDOM_STATE)


# ─────────────────────── File paths ────────────────────────────
DATA_FILE = "data/gender.csv"

# Google Drive persistence paths (set these when running on Colab)
# Example: DRIVE_ROOT = "/content/drive/MyDrive/NLP_project"
DRIVE_ROOT = "drive_output"            # local fallback
MODELS_DIR = f"{DRIVE_ROOT}/models"
RESULTS_DIR = f"{DRIVE_ROOT}/results"
LLM_CACHE_DIR = f"{DRIVE_ROOT}/llm_cache"

# ─────────────────────── Cross-Validation ──────────────────────
N_FOLDS = 5

# ─────────────────────── Text conditions ───────────────────────
TEXT_CONDITIONS = ["original", "structural", "random", "topical"]
TEXT_COLUMNS = {
    "original":   "text",
    "structural":  "text_structural",
    "random":      "text_random",
    "topical":     "text_topical",
}

# ─────────────────────── Lexical resource ──────────────────────
# Canonical base pairs from Bolukbasi et al. (2016, NeurIPS),
# Caliskan et al. (2017, Science), and Garg et al. (2018, PNAS).
_GENDER_PAIRS = [
    # (masculine, feminine)
    ("he", "she"),
    ("him", "her"),
    ("his", "her"),
    ("his", "hers"),
    ("himself", "herself"),
    ("boy", "girl"),
    ("brother", "sister"),
    ("father", "mother"),
    ("guy", "gal"),
    ("male", "female"),
    ("man", "woman"),
    ("nephew", "niece"),
    ("son", "daughter"),
    ("uncle", "aunt"),
    ("husband", "wife"),
    ("girlfriend", "boyfriend"),
    # Plural forms
    ("boys", "girls"),
    ("brothers", "sisters"),
    ("fathers", "mothers"),
    ("males", "females"),
    ("men", "women"),
    ("nephews", "nieces"),
    ("sons", "daughters"),
    ("uncles", "aunts"),
]

# Flatten into a single lookup set of lowercased lemmas for masking.
STRUCTURAL_GENDER_TERMS: set[str] = set()
for m, f in _GENDER_PAIRS:
    STRUCTURAL_GENDER_TERMS.add(m.lower())
    STRUCTURAL_GENDER_TERMS.add(f.lower())

# ─────────────────────── LLM evaluation ───────────────────────
LLM_SUBSAMPLE_N = 200         # stratified sample per fold
LLM_TEMPERATURE = 0.0
LLM_PROMPT = (
    "You are a computational sociolinguist. Based solely on the writing "
    "style of the following Reddit post, classify the likely gender of the "
    "author. Respond with exactly one word: Male or Female. Do not provide "
    "any explanation.\n\n"
    "Post:\n\"\"\"\n{text}\n\"\"\"\n\n"
    "Gender:"
)