# ⚽ Advanced Football Match Predictor (Machine Learning)

This Streamlit application provides football match predictions using a Machine Learning model. Unlike simpler models based only on odds, this predictor leverages historical data to make more informed predictions.

The application uses two `GradientBoostingClassifier` models trained on over 5,000 matches from the `ml_dataset_CLEAN.csv` dataset.

## Features Used in the Model

The prediction models use the following features:
- **Betting Odds**: Home Win, Draw, and Away Win odds.
- **Seasonal Performance**: The average number of goals scored per game by the home and away teams during the current season.
- **Head-to-Head (H2H) Stats**: The average number of goals scored in previous matches between the two teams.

## Installation

1.  Clone this repository or download the files (`prediction_model.py`, `train_model.py`, `ml_dataset_CLEAN.csv`, `requirements.txt`).
2.  Install the necessary Python packages using the `requirements.txt` file:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Train the Models (First-time setup)**:
    Before running the app for the first time, you need to train the machine learning models using the provided data. Run the training script from your terminal:
    ```bash
    python train_model.py
    ```
    This will create two files: `hda_model.joblib` and `ou_model.joblib`. You only need to do this once.

2.  **Run the Streamlit Application**:
    Once the models are trained, you can start the web application:
    ```bash
    streamlit run prediction_model.py
    ```

3.  **Get Predictions**:
    - Your web browser will open with the application running.
    - In the sidebar, select the **Home Team** and **Away Team** from the dropdown menus.
    - Enter the current **Match Odds**.
    - Click the **"Get Prediction"** button.
    - The application will look up the latest stats for the selected teams, combine them with the odds you provided, and show you the model's predictions and confidence levels.
