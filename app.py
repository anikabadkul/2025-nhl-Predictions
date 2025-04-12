import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Load saved model and scaler
model = joblib.load("nhl_cup_model.joblib")
scaler = joblib.load("scaler.joblib")

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
    "FaceoffWin%": [50.3, 50.3, 52.1, 49.2, 47.1, 50.0, 52.5, 53.8, 51.2, 49.1]
})

# Scale features
features = ["GoalsPerGame", "GoalsAgainstPerGame", "PowerPlay%", "PenaltyKill%", "FaceoffWin%"]
scaled_data = scaler.transform(data[features])
data["PredictedCupScore"] = model.predict(scaled_data)

# Sort for plotting
sorted_data = data.sort_values("PredictedCupScore", ascending=True)

# Streamlit UI
st.title("🏆 Predicted 2025 Stanley Cup Rankings")
st.write("Based on real 2024–25 NHL regular season team stats")

# Bar Chart
fig, ax = plt.subplots()
ax.barh(sorted_data["Team"], sorted_data["PredictedCupScore"])
ax.invert_yaxis()
ax.set_xlabel("Predicted Cup Score (lower is better)")
ax.set_title("2025 Stanley Cup Prediction")

st.pyplot(fig)

# Optional: Show raw data
if st.checkbox("Show team data"):
    st.dataframe(data)
