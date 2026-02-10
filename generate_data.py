import numpy as np
import pandas as pd

# Set random seed so results are same every time
np.random.seed(42)

# Number of fake people
N = 5000

# Generate features
age = np.random.randint(18, 80, N)
bmi = np.random.uniform(16, 40, N).round(1)
sleep_hours = np.random.uniform(3, 9, N).round(1)
stress_level = np.random.randint(1, 11, N)
exercise_days = np.random.randint(0, 8, N)

diabetes = np.random.choice([0, 1], N, p=[0.85, 0.15])
pcos = np.random.choice([0, 1], N, p=[0.9, 0.1])

diet = np.random.choice(
    ["Veg", "Mixed", "Non-Veg"],
    N,
    p=[0.3, 0.4, 0.3]
)

# Create dataframe
df = pd.DataFrame({
    "age": age,
    "bmi": bmi,
    "sleep_hours": sleep_hours,
    "stress_level": stress_level,
    "exercise_days": exercise_days,
    "diabetes": diabetes,
    "pcos": pcos,
    "diet_type": diet
})

# Save to CSV
df.to_csv("data/health_data_raw.csv", index=False)

print("Dataset created and saved to data/health_data_raw.csv")
