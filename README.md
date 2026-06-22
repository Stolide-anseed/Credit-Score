## Logistic regression experiment pipeline

This project has a lightweight experiment pipeline for checking preprocessing
hypotheses on already prepared feature files.

Input contract:

- CSV file contains ready-to-model features plus the target column.
- Feature columns are numeric and have no missing values.
- The target column has at least two classes.

Run:

```powershell
.\.venv\Scripts\python.exe -m model.experiment_pipeline path\to\features.csv --target Credit_Score --folds 5
```

Metrics printed per fold and as mean/std:

- precision macro/weighted
- recall macro/weighted
- ROC-AUC one-vs-rest macro/weighted
- average precision macro/weighted
- out-of-fold per-class precision, recall, ROC-AUC, average precision
- PR-curve point counts per class

To render PR curves locally:

```powershell
.\.venv\Scripts\python.exe -m model.experiment_pipeline path\to\features.csv --target Credit_Score --show-pr-curve
```
