# File paths
DATA_FILE = "data/gender.csv"

# Data split
TEST_SIZE = 0.3
RANDOM_STATE = 42

# Gender terms for masking
GENDER_TERMS = {
    "he", "she", "him", "her", "his", "hers",
    "husband", "wife", "boyfriend", "girlfriend",
    "bf", "gf", "man", "woman", "men", "women",
    "male", "female", "pregnant", "mom", "dad",
    "baby", "child", "children"
}

# Masking tokens
LEAKAGE_TOKEN = "[LEAKAGE]"
CONTENT_TOKEN = "[CONTENT]"