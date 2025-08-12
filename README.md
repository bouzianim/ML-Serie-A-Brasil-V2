# ⚽ Football Match Predictor

This Streamlit application provides football match predictions based on betting odds. It uses a Poisson distribution model to estimate expected goals (xG) from the odds and then simulates match outcomes to predict the final result.

The model's parameters have been optimized based on a comprehensive analysis of over 5,000 matches.

## Installation

1.  Clone this repository or download the files.
2.  Install the necessary Python packages using the `requirements.txt` file:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

You can run the prediction model as a Streamlit web application.

1.  Navigate to the directory containing the files in your terminal.
2.  Run the following command:

    ```bash
    streamlit run prediction_model.py
    ```

3.  Your web browser will open with the application running.
4.  Use the sidebar to input the odds for a home win, a draw, and an away win.
5.  Click the "Get Prediction" button to see the results.

## Understanding the Output

The application will display the predictions in a user-friendly format:

-   **Primary Predictions**: The main predictions for the Home/Draw/Away market and the Over/Under 2.5 goals market, along with the model's confidence percentage.
-   **High-Accuracy Pattern Signals**: This section highlights if any special, high-confidence patterns were detected in the odds. These signals indicate scenarios where the model has historically shown higher accuracy.
