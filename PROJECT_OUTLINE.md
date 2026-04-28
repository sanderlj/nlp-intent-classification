# NLP Assignment 1 Project Outline

## Goal

Build and compare text classification pipelines for intent detection in two African languages from the INJONGO dataset.

## Deliverables

1. `bpe.py` - standalone BPE tokenizer implementation
2. `nb.py` - standalone multinomial Naive Bayes implementation
3. `main.py` - experiment runner
4. `report.pdf` - 4-page report with analysis
5. `requirements.txt` - Python dependencies

## Folder Structure

```text
Assignment 1/
  data/
    raw/
    processed/
  outputs/
    metrics/
    plots/
    tokenization_examples/
    error_analysis/
  src/
    __init__.py
    config.py
    data_utils.py
    features.py
    evaluation.py
    experiments.py
  bpe.py
  nb.py
  main.py
  requirements.txt
  PROJECT_OUTLINE.md
```

## Work Plan

### 1. Language selection

- Choose two African languages from the assignment list
- Write down why they are interesting to compare
- Use the same two languages for all experiments

### 2. Data preparation

- Download the INJONGO intent dataset
- Keep the provided train/dev/test split unchanged
- Build a loader that returns texts and labels per split and language
- Inspect class counts and a few example utterances

### 3. Baselines

- Word unigram bag-of-words + Logistic Regression
- Character n-gram features + Logistic Regression
- Tune `C` on the dev set
- Tune character n-gram range on the dev set

### 4. BPE tokenizer

- Train separate BPE tokenizers for each language
- Support configurable merge count `k`
- Add end-of-word marker
- Save tokenization examples for `k = 100, 300, 500`

### 5. Naive Bayes

- Implement multinomial Naive Bayes from scratch
- Use BPE token counts as features
- Tune `alpha` and `k` jointly on the dev set

### 6. Feature-engineered Logistic Regression

- Use BPE features as the base
- Add at least two extra feature types
- Run ablations for every additional feature type

### 7. Report writing

- One summary table for all models and both languages
- One hyperparameter plot comparing both languages
- Tokenizer analysis with examples
- Error analysis for the best model
- Reflection on what features helped and why

## Suggested Tracking Table

For each run, record:

- language
- model
- features
- hyperparameters
- dev accuracy
- test accuracy
- notes

## Immediate Next Steps

1. Put the dataset into `data/raw/`
2. Choose two language codes in `src/config.py`
3. Implement the loader in `src/data_utils.py`
4. Run the baseline pipeline from `main.py`
