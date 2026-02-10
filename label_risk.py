import pandas as pd

# Load raw data
df = pd.read_csv("data/health_data_raw.csv")

# Function to decide risk level
def calculate_risk(row):
    score = 0

    # Age factor
    if row["age"] > 55:
        score += 2
    elif row["age"] > 40:
        score += 1

    # BMI factor
    if row["bmi"] > 32:
        score += 3
    elif row["bmi"] > 25:
        score += 2

    # Sleep factor
    if row["sleep_hours"] < 5:
        score += 2
    elif row["sleep_hours"] < 6:
        score += 1

    # Stress factor
    if row["stress_level"] >= 8:
        score += 2
    elif row["stress_level"] >= 5:
        score += 1

    # Exercise factor
    if row["exercise_days"] <= 1:
        score += 2
    elif row["exercise_days"] <= 3:
        score += 1

    # Diseases
    if row["diabetes"] == 1:
        score += 3

    if row["pcos"] == 1:
        score += 2

    # Final risk category
    if score >= 9:
        return "High"
    elif score >= 5:
        return "Medium"
    else:
        return "Low"


# Apply to dataset
df["risk_level"] = df.apply(calculate_risk, axis=1)

# Save new file
df.to_csv("data/health_data_labeled.csv", index=False)

print("Labeled dataset saved as data/health_data_labeled.csv")

# Show counts
print("\nRisk Level Distribution:")
print(df["risk_level"].value_counts())
