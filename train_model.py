import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


# 1. Load dataset
df = pd.read_csv("smart_crop_dataset.csv")

print("Dataset loaded successfully!")
print(df.head())


# 2. Separate input and output
X = df.drop("Irrigation", axis=1)
y = df["Irrigation"]


# 3. Define columns
categorical_columns = ["Crop_Type", "Growth_Stage"]

numerical_columns = [
    "Soil_Moisture",
    "Temperature",
    "Humidity",
    "Rainfall"
]


# 4. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ],
    remainder="passthrough"
)


# 5. Create model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)


# 6. Create complete pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# 7. Train model
pipeline.fit(X, y)


# 8. Save model
joblib.dump(pipeline, "irrigation_model.pkl")


print("✅ Model trained successfully!")
print("✅ irrigation_model.pkl created successfully!")