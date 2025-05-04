import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, RocCurveDisplay
)
from imblearn.over_sampling import SMOTE
from IPython.display import display 

def show_feature_importance(model, feat_names, top_n: int = 10,
                            title_prefix: str = "CatBoost"):
    """Print and plot the top-N feature importances."""
    importances = model.get_feature_importance(type="FeatureImportance")
    imp_series  = (pd.Series(importances, index=feat_names)
                   .sort_values(ascending=False))

    print(f"\nTop-{top_n} features by importance:")
    display(pd.DataFrame({"Feature": imp_series.index,
                          "Importance": imp_series.values}).head(top_n))

    plt.figure(figsize=(6, 4))
    imp_series.head(top_n)[::-1].plot(kind="barh")
    plt.title(f"{title_prefix} – Top-{top_n} Feature Importances")
    plt.xlabel("Importance (PredictionValuesChange)")
    plt.tight_layout()
    plt.show()

DATA_PATH = "1000.csv"
TARGET    = "Diagnosis"          # 0 = No Cancer, 1 = Cancer

df = pd.read_csv(DATA_PATH)

y = df[TARGET]
X = df.drop(columns=[TARGET])
feature_names = X.columns.tolist()

# CatBoost needs column indices (ints) for categoricals
cat_cols = [i for i, col in enumerate(X.columns) if col in ("Gender", "GeneticRisk")]

# 3. Train/val/test split  (64 % / 16 % / 20 %)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"[✓] Data shapes – train {X_train.shape}, test {X_test.shape}")

print("\nClass counts before SMOTE:", y_train.value_counts().to_dict())

param_grid = {
    "iterations":     [300, 600],
    "depth":          [4, 6],
    "learning_rate":  [0.03, 0.06],
    "l2_leaf_reg":    [1, 3],
}



cbc_base = CatBoostClassifier(
    loss_function="Logloss",
    eval_metric="F1",
    random_state=42,
    verbose=False
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(
    estimator=cbc_base,
    param_grid=param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1,
    verbose=1,
    refit=True
)

grid.fit(X_train, y_train, cat_features=cat_cols)

print("\nBest parameters ➜", grid.best_params_)
print(f"Cross-validated best F1: {grid.best_score_:.3f}")

val_pred = grid.predict(X_test)
print("Test performance:\n")
print(f"Accuracy : {accuracy_score(y_test, val_pred):.3f}")
print("Classification Matrix:")
print(classification_report(y_test, val_pred, digits=3))
print("Confusion Matrix:")
print(confusion_matrix(y_test, val_pred))
print(f"Validation – accuracy {accuracy_score(y_test, val_pred):.3f}, "
      f"F1 {f1_score(y_test, val_pred):.3f}")
ConfusionMatrixDisplay.from_predictions(y_test, val_pred, cmap="Blues")
plt.title("CatBoost Confusion Matrix (Test)")
plt.show()

# … your existing test-set code …
best_cbc = grid.best_estimator_
show_feature_importance(best_cbc, feature_names, top_n=10,
                        title_prefix="CatBoost (Test set)")


DATA_PATH = "500.csv"
TARGET    = "Diagnosis"

df_val = pd.read_csv(DATA_PATH)

y_val = df_val[TARGET]
X_val = df_val.drop(columns=[TARGET])
feature_names = X.columns.tolist()

cat_cols_val = [i for i, col in enumerate(X.columns) if col in ("Gender", "GeneticRisk")]

print(f"[✓] Data shapes – Validation dataset {X_val.shape}")

y_valid_pred = grid.predict(X_val)
print("Test performance:\n")
print(f"Accuracy : {accuracy_score(y_val, y_valid_pred):.3f}")
print("Classification Matrix:")
print(classification_report(y_val, y_valid_pred, digits=3))
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_valid_pred))
print(f"Validation – accuracy {accuracy_score(y_val, y_valid_pred):.3f}, "
      f"F1 {f1_score(y_val, y_valid_pred):.3f}")
ConfusionMatrixDisplay.from_predictions(y_val, y_valid_pred, cmap="Blues")
plt.title("CatBoost Confusion Matrix (Validation)")
plt.show()

# … your existing test-set code …

# … your existing validation-set code …

show_feature_importance(best_cbc, feature_names, top_n=10,
                        title_prefix="CatBoost (External validation)")


