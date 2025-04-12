import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib

# 2024–25 NHL regular season data
data = pd.DataFrame({
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
    "CupScore2025": [0.713059, 2.142619, 3.014304, 4.220603, 6.168551,
                     6.738807, 8.089947, 9.045675, 10.076142, 10.790294]
})

# Features and labels
X = data[["GoalsPerGame", "GoalsAgainstPerGame", "PowerPlay%", "PenaltyKill%", "FaceoffWin%"]]
y = data["CupScore2025"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LinearRegression()
model.fit(X_scaled, y)

# Save model and scaler
joblib.dump(model, "nhl_cup_model.joblib")
joblib.dump(scaler, "scaler.joblib")

print("Model and scaler saved successfully.")
