import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np

print("Explain script started...")

# -----------------------------
# Load model + data
# -----------------------------
model = joblib.load("models/random_forest_model.pkl")
print("Random Forest model loaded.")

X_train = joblib.load("models/X_train.pkl")
X_test = joblib.load("models/X_test.pkl")

# Convert sparse -> dense
if hasattr(X_train, "toarray"):
    X_train_dense = X_train.toarray()
else:
    X_train_dense = X_train

if hasattr(X_test, "toarray"):
    X_test_dense = X_test.toarray()
else:
    X_test_dense = X_test

print("Data loaded.")

preprocessor = joblib.load("models/preprocessor.pkl")

# -----------------------------
# Get EXACT feature names
# -----------------------------
feature_names = preprocessor.get_feature_names_out().tolist()

print("Feature names prepared:", len(feature_names))

# -----------------------------
# Create SHAP explainer
# -----------------------------
print("Creating SHAP explainer...")

masker = shap.maskers.Independent(X_train_dense)

explainer = shap.Explainer(model.predict_proba, masker)

shap_values = explainer(X_test_dense)

print("SHAP values computed.")

# -----------------------------
# GLOBAL explanation
# -----------------------------
print("Showing global feature importance...")

# Class 0 = Low risk
shap.summary_plot(
    shap_values.values[:, :, 0],
    X_test_dense,
    feature_names=feature_names,
    show=True,
)

# -----------------------------
# LOCAL explanation
# -----------------------------
print("Showing local explanation for one person...")

idx = 0

shap.waterfall_plot(
    shap.Explanation(
        values=shap_values.values[idx, :, 0],
        base_values=shap_values.base_values[idx, 0],
        data=X_test_dense[idx],
        feature_names=feature_names,
    )
)

plt.show()

print("Finished SHAP explanations.")
