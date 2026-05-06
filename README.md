# NLP Assignment 1 Project Outline

Course assignment project for multilingual intent classification on Amharic (`amh`) and Zulu (`zul`).

## What this repo contains

- `main.py` — entry point for running experiments
- `bpe.py` — standalone BPE tokenizer implementation
- `nb.py` — standalone multinomial Naive Bayes implementation
- `src/` — experiment, feature, evaluation, config, and reporting helpers
- `requirements.txt` — Python dependencies
- `outputs/` — generated metrics, plots, and analysis artifacts when running
- data expected in `data/raw/...` 

## Run

first ensure requirements, then run the main.py
```bash
python -m pip install -r requirements.txt
```

```bash
python main.py
