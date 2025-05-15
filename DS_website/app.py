from pathlib import Path
import os, json, joblib, tempfile, re, traceback

import numpy as np
import pandas as pd
from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS
from catboost import CatBoostClassifier, Pool

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

BASE_DIR      = Path(__file__).parent.resolve()
MODEL_PATH    = BASE_DIR / "model/best_cancer_model.cbm"
FEATURES_PATH = BASE_DIR / "model/feature_names.json"
CAT_COLS_PATH = BASE_DIR / "model/cat_cols.pkl"

THRESH = 0.50
MAP_SEX   = {"Male": 0, "Female": 1}
MAP_BOOL  = {"No": 0, "Yes": 1}
MAP_RISK  = {"Low": 0, "Medium": 1, "High": 2}

model = CatBoostClassifier()
model.load_model(str(MODEL_PATH))

FEATURE_NAMES = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
CAT_COLS_OBJ  = joblib.load(CAT_COLS_PATH)

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

@app.route("/")
def index(): return send_from_directory(".", "dataset1.html")
@app.route("/<path:p>")
def static_files(p): return send_from_directory(".", p)

@app.route("/predict", methods=["POST"])
def predict():
    """
    Payload: JSON array (the validated rows)
    Flow:
        1) save rows → temp.csv
        2) Selenium ⟶ dataset1.html ⟶ upload csv ⟶ scrape table
        3) clean → encode → CatBoost
        4) return enriched JSON
    """
    rows = request.get_json(force=True)
    if not isinstance(rows, list) or not rows:
        return jsonify({"error": "Expected non‑empty JSON array"}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", dir=str(BASE_DIR)) as tmp:
        csv_path = Path(tmp.name)
        pd.DataFrame(rows).to_csv(csv_path, index=False)

    try:
        scraped = _scrape_dataset(csv_path)
        if not scraped:
            return jsonify({"error": "Selenium could not read any rows"}), 500

        print(f"\n[DEBUG] Scraped {len(scraped)} rows; first row:")
        print(scraped[0], "\n")

        df_out = _score_dataframe(pd.DataFrame(scraped))
        return df_out.to_json(orient="records")

    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500

    finally:
        try: os.remove(csv_path)
        except OSError: pass

_RGX_NUM = re.compile(r"[^\d.\-]")

def _num(series):
    return (
        series.astype(str)
              .str.replace(_RGX_NUM, "", regex=True)
              .replace("", np.nan)
              .astype(float)
    )

def encode_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Cast every column to the exact dtype used at training."""
    df = df.copy()

    df["Gender"]         = df["Gender"].map(MAP_SEX).fillna(-1).astype("int8")
    df["Smoking"]        = df["Smoking"].map(MAP_BOOL).fillna(-1).astype("int8")
    df["Cancer History"] = df["Cancer History"].map(MAP_BOOL).fillna(-1).astype("int8")
    df["Genetic Risk"]   = df["Genetic Risk"].map(MAP_RISK).fillna(-1).astype("int8")

    df["Age"] = pd.to_numeric(df["Age"], errors="coerce").fillna(-1).astype("int8")
    df["BMI"] = pd.to_numeric(df["BMI"], errors="coerce")

    df["Physical Activity"] = _num(df["Physical Activity"])
    df["Alcohol Intake"]    = _num(df["Alcohol Intake"])

    for col in ["BMI", "Physical Activity", "Alcohol Intake"]:
        df[col] = df[col].astype("float32")

    return df

def _score_dataframe(df_disp: pd.DataFrame) -> pd.DataFrame:
    """clean → encode → predict; return DataFrame ready for UI."""
    df_enc = encode_frame(df_disp)

    X = df_enc[FEATURE_NAMES]
    cat_idx = [c if isinstance(c, (int, np.integer)) else X.columns.get_loc(c)
               for c in CAT_COLS_OBJ]

    pool   = Pool(X, cat_features=cat_idx)
    proba  = model.predict_proba(pool)[:, 1]
    pred   = (proba >= THRESH).astype(int)

    df_disp["BMI"]               = df_enc["BMI"].round(1)
    df_disp["Physical Activity"] = df_enc["Physical Activity"].round(1)
    df_disp["Alcohol Intake"]    = df_enc["Alcohol Intake"].round(1)

    df_disp["Cancer_Probability"] = proba
    df_disp["Cancer_Prediction"]  = pred
    return df_disp

def _scrape_dataset(csv_to_load: Path) -> list[dict]:
    """Return list‑of‑dicts scraped from the table after upload."""
    url = "file:///" + str((BASE_DIR / "dataset1.html").resolve()).replace("\\", "/")

    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1920,1200")
    opts.add_experimental_option("excludeSwitches", ["enable-logging"])

    driver = webdriver.Chrome(options=opts)
    try:
        driver.get(url)

        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "csv-file"))
        )
        driver.find_element(By.ID, "csv-file").send_keys(str(csv_to_load.resolve()))

        WebDriverWait(driver, 25).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "#data-table tbody tr"))
        )

        headers = [th.text.strip()
                   for th in driver.find_elements(By.CSS_SELECTOR, "#data-table thead th")]
        rows = []
        for r in driver.find_elements(By.CSS_SELECTOR, "#data-table tbody tr"):
            cells = [c.text.strip() for c in r.find_elements(By.TAG_NAME, "td")]
            if cells: rows.append(dict(zip(headers, cells)))
        return rows
    finally:
        driver.quit()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
