# Football Match Prediction Model

This script provides football match predictions based on betting odds. It uses a Poisson distribution model to estimate expected goals (xG) from the odds and then simulates match outcomes to predict the final result.

The model's parameters have been optimized based on a comprehensive analysis of over 5,000 matches.

## Installation

1.  Clone this repository or download the files.
2.  Install the necessary Python packages using the `requirements.txt` file:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

You can run the prediction model from the command line. You need to provide the odds for a home win, a draw, and an away win.

```bash
python prediction_model.py --home <home_odds> --draw <draw_odds> --away <away_odds>
```

### Example

```bash
python prediction_model.py --home 2.0 --draw 3.2 --away 3.5
```

## Understanding the Output

The script will output a JSON object with the following structure:

```json
{
    "primary_hda": {
        "prediction": "H",
        "confidence": "45.3%",
        "details": "Using Method 1 with 11 scores"
    },
    "primary_ou": {
        "prediction": "UNDER 2.5",
        "confidence": "64.6%",
        "details": "Using Method 1 with 17 scores"
    },
    "special_patterns": {
        "high_confidence_hda": {
            "triggered": false,
            "prediction": "N/A",
            "confidence": "N/A",
            "details": "Signal for H/D/A when confidence is >= 60% using 10 scores"
        },
        "high_avg_pattern_ou": {
            "triggered": false,
            "prediction": "N/A",
            "confidence": "N/A",
            "details": "Signal for OVER 2.5 when weighted goal average is >= 2.5 using 8 scores"
        }
    }
}
```

-   **primary_hda**: The main prediction for the Home/Draw/Away market.
-   **primary_ou**: The main prediction for the Over/Under 2.5 goals market.
-   **special_patterns**: These are checks for specific, high-accuracy scenarios:
    -   `high_confidence_hda`: This signal triggers if the model has a very high confidence (>=60%) in its H/D/A prediction.
    -   `high_avg_pattern_ou`: This signal triggers if the model finds a strong pattern suggesting an "Over 2.5" goals outcome.
