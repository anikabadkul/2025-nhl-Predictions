🏒 NHL Stanley Cup 2025 Prediction Dashboard

This project provides a machine learning pipeline and interactive dashboard to predict NHL teams’ chances of winning the 2025 Stanley Cup using data from the 2024–2025 regular season.

Project Files

1. `train_model.py`

A Python script that:

- Uses 2024–2025 NHL regular season team data
- Trains a machine learning model to estimate "Cup Score" — a proxy for championship potential
- Applies feature scaling to normalize inputs
- Saves:
  - Trained model → `nhl_cup_model.joblib`
  - Scaler → `nhl_scaler.joblib`

2. `prediction2.py`

A backend module that:

- Loads the trained model and scaler
- Accepts user/team inputs
- Processes and transforms data for prediction
- Returns the predicted Cup-winning score
- Used internally by the app

3. `app.py`

An interactive Streamlit app that:

- Loads the trained model and scaler
- Allows users to enter performance stats (GPG, PP%, PK%, etc.)
- Displays:
  - Real-time Cup Score prediction
  - Easy-to-use sliders and inputs
- Designed for local and Streamlit Cloud deployment

4. `nhl_cup_model.joblib` & `nhl_scaler.joblib`

- Trained artifacts created using `train_model.py`
- Required by `app.py` for accurate, instant predictions

Key Metrics Explained

- **Cup Score**: A model-derived score that reflects a team’s potential to win the Stanley Cup, based on 2024–2025 season performance stats

How to Run

Locally

1. Train or update the model (optional):

   ```bash
   python train_model.py
   ```

2. Launch the dashboard:

   ```bash
   streamlit run app.py
   ```

🛠️ Technologies Used

- Python 3
- `pandas`, `numpy` for data handling
- `scikit-learn` for modeling & scaling
- `joblib` for model persistence
- `streamlit` for the interactive web app

Future Plans

- Update model with 2025–2026 season data
- Add real-time stats scraping via API
- Visualize feature importance and trends inside the dashboard
