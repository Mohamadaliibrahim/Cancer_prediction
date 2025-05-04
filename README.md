# Patient Cancer‑Risk Prediction with CatBoost + SMOTE

## Overview

This project builds a **binary‑classification** model that predicts whether a patient is diagnosed with cancer (1) or not (0) based on simple demographic and lifestyle attributes.
We adopt **CatBoost Gradient Boosting** for its excellent performance on tabular data and native handling of categorical columns. To mitigate class imbalance we oversample the minority class with **SMOTE**.

## Dataset

| Column                 | Type        | Range / Categories            |
| ---------------------- | ----------- | ----------------------------- |
| `Age`                  | Integer     | 20 – 80                       |
| `Gender`               | Categorical | 0 (Male), 1 (Female)          |
| `BMI`                  | Continuous  | 15 – 40                       |
| `Smoking`              | Binary      | 0 (No), 1 (Yes)               |
| `GeneticRisk`          | Categorical | 0 (Low), 1 (Medium), 2 (High) |
| `PhysicalActivity`     | Continuous  | 0 – 10 hours/week             |
| `AlcoholIntake`        | Continuous  | 0 – 5 units/week              |
| `CancerHistory`        | Binary      | 0 (No), 1 (Yes)               |
| `Diagnosis` *(target)* | Binary      | 0 (No Cancer), 1 (Cancer)     |

The working CSVs live under `data/` (e.g. `1400.csv`, `100.csv`).

## Repository Structure

```
.
├── notebooks/
│   └── catboost_smote_pipeline.ipynb   # main Colab / Jupyter notebook
├── data/
│   ├── 1400.csv                        # training / test split source
│   └── 100.csv                         # external validation set
├── README.md                           # you are here
└── requirements.txt                    # python dependencies
```

## Requirements

* Python ≥ 3.8
* catboost ≥ 1.2
* scikit‑learn ≥ 1.4
* imbalanced‑learn ≥ 0.12
* pandas, matplotlib

Install everything in one shot:

```bash
pip install -r requirements.txt
```

## Quick Start (Colab)

1. Open **`notebooks/catboost_smote_pipeline.ipynb`** in Google Colab.
2. Mount Google Drive and confirm the `DATA_PATH` variables point to your CSV files.
3. Run **all cells**.
   *The notebook automatically*:

   * splits data (80 % train, 20 % test),
   * applies **SMOTE** to the training fold only,
   * performs **GridSearchCV** hyper‑parameter tuning,
   * evaluates on the untouched test set & external validation set, and
   * prints / plots feature importances.

## Key Steps in the Pipeline

| Stage               | Description                                                                         |
| ------------------- | ----------------------------------------------------------------------------------- |
| **1 – Split**       | Stratified train/test split to preserve class ratios.                               |
| **2 – SMOTE**       | Oversample minority class in **train** only to avoid data leakage.                  |
| **3 – Grid Search** | Tune `iterations`, `depth`, `learning_rate`, `l2_leaf_reg` on 5‑fold CV (F1 score). |
| **4 – Fit**         | Re‑train best CatBoost on oversampled Train + Val data.                             |
| **5 – Evaluate**    | Accuracy, F1, confusion matrix, ROC curve on held‑out Test + external Validation.   |
| **6 – Explain**     | `get_feature_importance()` + bar chart of top 10 predictors.                        |

## Baseline Results

| Dataset                                                          |   Accuracy |   F1 Score |
| ---------------------------------------------------------------- | ---------: | ---------: |
| Test (20 %)                                                      | ≈ **0.90** | ≈ **0.89** |
| External Valid.                                                  | ≈ **0.88** | ≈ **0.87** |
| *Exact numbers vary with random seed and future data additions.* |            |            |

## Using the Model for New Patients

```python
from pipeline.inference import predict_on_new_csv  # utility defined in notebook
preds = predict_on_new_csv("data/new_patients.csv")
print(preds.head())
```

The helper enforces column order and returns a DataFrame with a `Prediction` (0/1) column.

## License

MIT License © 2025 Mhmd Ali
