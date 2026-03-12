# Multi-Label Scientific Paper Classification

This project implements a full machine learning pipeline for multi-label classification of scientific papers.

## Problem
Given TITLE and ABSTRACT, predict one or more labels from:
- Computer Science
- Physics
- Mathematics
- Statistics
- Quantitative Biology
- Quantitative Finance

## What is implemented
- EDA in notebook
- Configurable text preprocessing pipeline
- Feature extraction with:
  - TF-IDF (traditional)
  - BERT embeddings via SentenceTransformers (`all-MiniLM-L6-v2`)
- Multi-label training with One-vs-Rest:
  - Logistic Regression
  - Naive Bayes
  - LinearSVC
- Evaluation with Micro-F1 and Macro-F1

## Repository structure

```text
.
|-- main.ipynb
|-- src/
|   |-- modules/
|   |   |-- preprocessing.py
|   |   |-- feature_extraction.py
|   |   |-- models.py
|   |   `-- evaluation.py
|   |-- data/        # generated train/val splits
|   |-- features/    # generated TF-IDF/BERT features
|   `-- models/      # generated trained models
`-- report/
  |-- 1 EDA.md
  |-- 2*.md
  |-- 3*.md
  |-- 4*.md
  `-- 5*.md
```

## Environment setup

1. Create and activate virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -U pip
pip install pandas numpy scipy scikit-learn matplotlib seaborn wordcloud sentence-transformers jupyter
```

## Run
Open and run all cells in [main.ipynb](main.ipynb).

Expected pipeline flow:
1. Load and inspect data
2. Preprocess text and split train/validation
3. Build TF-IDF and BERT features
4. Train 6 model combinations
5. Evaluate and compare metrics

## Current benchmark (validation)

| Feature | Model | Micro-F1 | Macro-F1 |
|---|---|---:|---:|
| TF-IDF | LinearSVC | 0.8254 | 0.7071 |
| TF-IDF | NaiveBayes | 0.8193 | 0.7364 |
| BERT | LinearSVC | 0.8189 | 0.7131 |
| TF-IDF | LogisticRegression | 0.8168 | 0.5671 |
| BERT | LogisticRegression | 0.8149 | 0.6801 |
| BERT | NaiveBayes | 0.7831 | 0.6897 |

## Notes for GitHub
- Generated artifacts are ignored by [.gitignore](.gitignore): split CSVs, feature arrays, trained model binaries.
- Keep dataset local (`train.csv` ignored) and document its source if needed.
- Commit notebook, source modules, and reports.

## Suggested first commit

```powershell
git init
git add .
git commit -m "Initial ML pipeline: preprocessing, features, models, evaluation, reports"
```
