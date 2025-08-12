import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Load Models and Data ---
@st.cache_data
def load_data_and_models():
    """Load the dataset and the trained machine learning models."""
    df = pd.read_csv('ml_dataset_CLEAN.csv')
    # Convert date to datetime for sorting
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%y')
    df.sort_values(by='Date', ascending=False, inplace=True)

    hda_model = joblib.load('hda_model.joblib')
    ou_model = joblib.load('ou_model.joblib')

    # Get a unique list of teams for the dropdowns
    all_teams = pd.concat([df['Home'], df['Away']]).unique()
    all_teams.sort()

    return df, hda_model, ou_model, all_teams

df, hda_model, ou_model, all_teams = load_data_and_models()

# --- Streamlit App UI ---
st.set_page_config(page_title="⚽ Advanced Match Predictor", layout="wide")
st.title("🎯 Advanced Football Match Predictor (ML)")
st.markdown("Select teams and enter odds to get predictions from a Machine Learning model trained on historical data.")

with st.sidebar:
    st.header("Match Details")

    home_team = st.selectbox("Home Team", options=all_teams, index=0)
    away_team = st.selectbox("Away Team", options=all_teams, index=len(all_teams) - 1)

    st.header("Match Odds")
    odds_home = st.number_input("Home Win Odds", min_value=1.01, value=2.5, step=0.1)
    odds_draw = st.number_input("Draw Odds", min_value=1.01, value=3.2, step=0.1)
    odds_away = st.number_input("Away Win Odds", min_value=1.01, value=2.8, step=0.1)

    predict_button = st.button("🔮 Get Prediction", use_container_width=True)

if predict_button:
    if home_team == away_team:
        st.error("Home and Away teams cannot be the same. Please select different teams.")
    else:
        with st.spinner('🧠 Analyzing teams and running ML predictions...'):
            # --- Feature Fetching Logic ---
            try:
                # Get latest seasonal stats for home team
                latest_home_stats = df[df['Home'] == home_team].iloc[0]
                home_goals_season = latest_home_stats['Home_Team_Goals_Per_Game_Season']

                # Get latest seasonal stats for away team
                latest_away_stats = df[df['Away'] == away_team].iloc[0]
                away_goals_season = latest_away_stats['Away_Team_Goals_Per_Game_Season']

                # Get H2H stats
                h2h_df = df[((df['Home'] == home_team) & (df['Away'] == away_team)) |
                            ((df['Home'] == away_team) & (df['Away'] == home_team))]

                if not h2h_df.empty:
                    h2h_goals = h2h_df.iloc[0]['H2H_Goals_Per_Game']
                else:
                    # If no H2H data, use the average of their seasonal goals as a fallback
                    h2h_goals = (home_goals_season + away_goals_season) / 2
                    st.warning("No direct Head-to-Head match data found. Using a fallback estimate for H2H stats.")

                # --- Prediction ---
                # Assemble feature vector
                features = np.array([[
                    odds_home, odds_draw, odds_away,
                    home_goals_season,
                    away_goals_season,
                    h2h_goals
                ]])

                # Make predictions
                hda_prediction = hda_model.predict(features)[0]
                hda_probabilities = hda_model.predict_proba(features)[0]

                ou_prediction_val = ou_model.predict(features)[0]
                ou_prediction = "OVER 2.5" if ou_prediction_val == 1 else "UNDER 2.5"
                ou_probabilities = ou_model.predict_proba(features)[0]

                # --- Display Results ---
                st.subheader("📈 ML Model Predictions")
                col1, col2 = st.columns(2)

                with col1:
                    hda_confidence = hda_probabilities[np.where(hda_model.classes_ == hda_prediction)][0]
                    st.metric(
                        label="Match Outcome (H/D/A)",
                        value=hda_prediction,
                        delta=f"{hda_confidence*100:.1f}% Confidence"
                    )

                with col2:
                    ou_confidence = ou_probabilities[np.where(ou_model.classes_ == ou_prediction_val)][0]
                    st.metric(
                        label="Goal Total (Over/Under 2.5)",
                        value=ou_prediction,
                        delta=f"{ou_confidence*100:.1f}% Confidence"
                    )

                # Expander for more details
                with st.expander("Show detailed probabilities and features"):
                    st.write("**Input Features Used for Prediction:**")
                    st.json({
                        'Odd Home': odds_home,
                        'Odd Draw': odds_draw,
                        'Odd Away': odds_away,
                        'Home Team Goals/Game (Season)': f"{home_goals_season:.2f}",
                        'Away Team Goals/Game (Season)': f"{away_goals_season:.2f}",
                        'H2H Goals/Game': f"{h2h_goals:.2f}"
                    })

                    st.write("**H/D/A Probabilities:**")
                    st.json({cls: f"{prob*100:.1f}%" for cls, prob in zip(hda_model.classes_, hda_probabilities)})

                    st.write("**O/U 2.5 Probabilities:**")
                    st.json({("OVER 2.5" if cls == 1 else "UNDER 2.5"): f"{prob*100:.1f}%" for cls, prob in zip(ou_model.classes_, ou_probabilities)})

            except IndexError:
                st.error("Could not find sufficient historical data for one or both of the selected teams. Please try different teams.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
