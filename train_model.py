"""
train_model.py
===============
Retrains the irrigation prediction model on a real dataset and saves
it as irrigation_model.pkl in the same format the app expects: a
scikit-learn Pipeline with two named steps, "preprocessor" and "model".

WHY RETRAIN
-----------
If your current irrigation_model.pkl was trained on a small or
synthetic dataset, retraining on a larger real-world dataset (e.g. a
crop irrigation dataset from Kaggle) and trying a stronger model such
as RandomForest or XGBoost usually improves accuracy and makes the
SHAP explanations more meaningful.

HOW TO USE
----------
1. Get a real irrigation/crop dataset as a CSV with at least these
   columns (rename your columns to match, or edit COLUMN_MAP below):
       Soil_Moisture, Temperature, Humidity, Rainfall,
       Crop_Type, Growth_Stage, Irrigation_Required (0 or 1)

   A few public options to search for:
     - Kaggle: "Crop irrigation scheduling dataset"
     - Kaggle: "Smart irrigation system dataset"
     - Indian government open data portals (data.gov.in) for
       agriculture / soil moisture datasets

2. Place the CSV next to this script as: dataset.csv

3. Run:
       pip install -r requirements.txt
       python train_model.py

4. This overwrites irrigation_model.pkl. Test locally with
   `streamlit run app.py` before redeploying.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
import joblib

# ------------------------------------------------------------------
# EDIT THIS if your CSV uses different column names
# ------------------------------------------------------------------
COLUMN_MAP = {
    "Soil_Moisture": "Soil_Moisture",
    "Temperature": "Temperature",
    "Humidity": "Humidity",
    "Rainfall": "Rainfall",
    "Crop_Type": "Crop_Type",
    "Growth_Stage": "Growth_Stage",
    "Irrigation_Required": "Irrigation_Required",
}

NUMERIC_FEATURES = ["Soil_Moisture", "Temperature", "Humidity", "Rainfall"]
CATEGORICAL_FEATURES = ["Crop_Type", "Growth_Stage"]
TARGET = "Irrigation_Required"


def load_dataset(path="dataset.csv"):
    df = pd.read_csv(path)
    df = df.rename(columns={v: k for k, v in COLUMN_MAP.items()})
    missing = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}. "
            f"Update COLUMN_MAP in train_model.py to match your CSV."
        )
    return df


def build_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    return pipeline


def main():

    print("Loading dataset...")
    df = load_dataset()
    print(f"Loaded {len(df)} rows.")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()

    print("Running 5-fold cross-validation...")
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")
    print(f"Cross-val accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

    print("Training final model on the full training split...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    print("\n=== Test Set Performance ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(pipeline, "irrigation_model.pkl")
    print("\nSaved retrained model to irrigation_model.pkl")


if __name__ == "__main__":
    main()
