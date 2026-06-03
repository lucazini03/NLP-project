# Reddit Gender Classification with Data Depollution

This repository contains the **fully reproducible code** for the paper:

> *Unmasking Style: Evaluating Text Distortion for Data Depollution in Reddit Author Profiling*.

The goal of this project is to **quantitatively and qualitatively evaluate data pollution** in Reddit-based gender prediction, and to assess whether **stylometric signals remain predictive after masking label-aligned and topical content**.

---

## Repository Structure

config.py Global settings and constants

prepare_data.py Data loading, author-disjoint splitting, deduplication

masking.py Gender masking and POS-based noun masking

train_models.py Model training and evaluation

qualitative.py Feature importance and qualitative analysis

main.py End-to-end experiment pipeline


---

## Requirements

**Python:** 3.9+ (tested on Python 3.10)

### Dependencies

```bash
pip install pandas numpy scikit-learn spacy tqdm
python -m spacy download en_core_web_sm
```

## Data

- **Input file:** `gender.csv`
- Each row corresponds to a **single Reddit post**
- Add gender.csv to data directory

Reproducing the Experiments

Run the full pipeline with:
```bash
python main.py
```
This may take **several hours** to run due to POS tagging over the full dataset.

The pipeline performs the following steps:
1. Loads and cleans the data  
2. Splits the data **by author** (no author overlap between train and test)  
3. Applies masking strategies  
4. Trains all models  
5. Produces quantitative and qualitative outputs  

## Experimental Conditions

Each model is evaluated under **three preprocessing conditions**:

1. **Original text**
2. **Gender-masked text**  
   (explicit gender-related terms → `[LEAKAGE]`)
3. **POS-masked text**  
   (all nouns and proper nouns → `[CONTENT]`)

### Models
- Multinomial Naive Bayes  
- Logistic Regression  
- Linear Support Vector Machine  

### Metrics
- Accuracy  
- Macro-F1  

---

## Output Files and Paper Mapping

| File                    | Description                              | 
|-------------------------|------------------------------------------|
| `results.csv`           | Accuracy and Macro-F1 scores              | 
| `original_features.csv` | Top features learned on raw text          |
| `masked_features.csv`   | Top features learned on masked text       | 
| `feature_table.tex`     | LaTeX-ready feature table                 | 

All reported results in the paper are **directly generated** by this codebase.

---

## Reproducibility Notes

- All experiments use a **fixed random seed** (defined in `config.py`)
- Masking rules and feature selection are derived **only from training data**
- No test data is used during preprocessing or analysis

Minor numerical differences (±0.5%) may occur due to differences in hardware or library versions.

---

## Modifying or Extending the Code

- **Change masking rules:** edit `masking.py`
- **Add or modify models:** extend `train_models.py`
- **Experiment with different splits:** adjust `prepare_data.py`
- **Change hyperparameters or paths:** update `config.py`

The pipeline is modular and designed to support controlled extensions.

---

## Common Issues

- **spaCy model not found**  
  → Run `python -m spacy download en_core_web_sm`

- **Results differ slightly from the paper**  
  → Check the random seed and dataset version

- **Slow runtime**  
  → Disable POS masking for quick sanity checks



