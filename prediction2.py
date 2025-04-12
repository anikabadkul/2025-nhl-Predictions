# prediction2.py (Train on 2024-25 NHL stats and save model)
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib

# Real 2024-25 NHL regular season stats
train_data = pd.DataFrame({
    "Team": [
        "Tampa Bay Lightning", "Washington Capitals", "Dallas Stars", "Winnipeg Jets",
        "Colorado Avalanche", "Vegas Golden Knights", "Carolina Hurricanes",
        "Toronto Maple Leafs", "Edmonton Oilers", "Florida Panthers"
    ],
    "GoalsPerGame": [3.56, 3.62, 3.43, 3.42, 3.35, 3.35, 3.28, 3.25, 3.23, 3.23],
    "GoalsAgainstPerGame": [2.61, 2.68, 2.51, 2.32, 2.95, 2.96, 2.70, 2.90, 3.12, 2.47],
    "PowerPlay%": [25.1, 22.8, 22.1, 29.4, 25.1, 30.4, 19.0, 26.1, 25.5, 19.0],
    "PenaltyKill%": [82.1, 81.3, 84.9, 79.9, 79.8, 75.2, 84.6, 78.1, 77.2, 84.6],
    "FaceoffWin%": [50.3, 50.3, 52.1, 49.2, 47.1, 50.0, 52.5, 53.8, 51.2, 49.1],
    "StanleyCupPoints": [1, 2, 3, 4, 6, 7, 8, 9, 10, 11]  # dummy Cup performance ranking
})

features = ["GoalsPerGame", "GoalsAgainstPerGame", "PowerPlay%", "PenaltyKill%", "FaceoffWin%"]
X = train_data[features]
y = train_data["StanleyCupPoints"]

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train
model = LinearRegression()
model.fit(X_scaled, y)

# Predict and evaluate
train_data["PredictedCupScore"] = model.predict(X_scaled)
mape = abs(train_data["PredictedCupScore"] - y).mean()

# Sort
sorted_df = train_data[["Team", "PredictedCupScore"]].sort_values("PredictedCupScore").reset_index(drop=True)

# Save model + scaler
joblib.dump(model, "nhl_cup_model.joblib")
joblib.dump(scaler, "nhl_scaler.joblib")

# Output
print("\nPredicted 2025 Stanley Cup Rankings based on 2024–25 Regular Season Stats:")
print(sorted_df.to_string(index=False))
print(f"\nModel Error (MAE): {mape:.2f} points")



# Save the trained scaler and model
joblib.dump(scaler, "scaler.joblib")
joblib.dump(model, "nhl_cup_model.joblib")

