import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import joblib

print("Loading labeled dataset...")

df = pd.read_csv("data/health_data_labeled.csv")

# Features and target
X = df.drop("risk_level", axis=1)
y = df["risk_level"]

# Encode target labels (Low, Medium, High -> 0,1,2)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

joblib.dump(label_encoder, "models/label_encoder.pkl")

# Column types
numeric_features = [
    "age", "bmi", "sleep_hours",
    "stress_level", "exercise_days",
    "diabetes", "pcos"
]

categorical_features = ["diet_type"]

# Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# Fit preprocessor ONLY on training data
print("Fitting preprocessor...")
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Save processed arrays
joblib.dump(preprocessor, "models/preprocessor.pkl")
joblib.dump(X_train_processed, "models/X_train.pkl")
joblib.dump(X_test_processed, "models/X_test.pkl")
joblib.dump(y_train, "models/y_train.pkl")
joblib.dump(y_test, "models/y_test.pkl")

print("Preprocessing complete and files saved in models/ folder.")
