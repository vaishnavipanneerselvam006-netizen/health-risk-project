import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("Loading processed data...")

X_train = joblib.load("models/X_train.pkl")
X_test = joblib.load("models/X_test.pkl")
y_train = joblib.load("models/y_train.pkl")
y_test = joblib.load("models/y_test.pkl")

# --------------------
# 1) Logistic Regression
# --------------------
print("\nTraining Logistic Regression...")
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)
print("\nLogistic Regression Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

joblib.dump(lr, "models/logistic_model.pkl")

# --------------------
# 2) Random Forest
# --------------------
print("\nTraining Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
print("\nRandom Forest Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

joblib.dump(rf, "models/random_forest_model.pkl")

# --------------------
# 3) Gradient Boosting
# --------------------
print("\nTraining Gradient Boosting...")
gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)

y_pred_gb = gb.predict(X_test)
print("\nGradient Boosting Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_gb))
print(classification_report(y_test, y_pred_gb))

joblib.dump(gb, "models/gradient_boosting_model.pkl")

print("\nAll models trained and saved!")
