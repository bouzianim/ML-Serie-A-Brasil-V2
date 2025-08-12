import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
df = pd.read_csv('ml_dataset_CLEAN.csv')

# --- 1. Data Exploration ---
print("--- Dataset Info ---")
df.info()

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Statistical Summary ---")
print(df.describe())

# --- 2. Feature Engineering and Data Preparation ---
print("\n--- Feature Engineering ---")

# Create target variable for H/D/A prediction
def get_result(row):
    if row['HG'] > row['AG']:
        return 'H'
    elif row['HG'] < row['AG']:
        return 'A'
    else:
        return 'D'
df['Result'] = df.apply(get_result, axis=1)

# Create target variable for O/U 2.5 prediction
df['Over_Under_2.5'] = (df['HG'] + df['AG'] > 2.5).astype(int)

# Feature Selection
features = [
    'Odd Home', 'Odd Draw', 'Odd Away',
    'Home_Team_Goals_Per_Game_Season',
    'Away_Team_Goals_Per_Game_Season',
    'H2H_Goals_Per_Game'
]

# Drop rows with missing values in our features or target
df.dropna(subset=features + ['Result', 'Over_Under_2.5'], inplace=True)

# Define features (X) and targets (y)
X = df[features]
y_hda = df['Result']
y_ou = df['Over_Under_2.5']

print(f"Selected features: {features}")
print(f"Number of samples after cleaning: {len(X)}")

# --- 3. Model Training ---
print("\n--- Model Training ---")

# Split data for H/D/A model
X_train_hda, X_test_hda, y_train_hda, y_test_hda = train_test_split(X, y_hda, test_size=0.2, random_state=42)

# Train H/D/A model
print("Training H/D/A model...")
hda_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
hda_model.fit(X_train_hda, y_train_hda)
y_pred_hda = hda_model.predict(X_test_hda)
accuracy_hda = accuracy_score(y_test_hda, y_pred_hda)
print(f"H/D/A Model Accuracy: {accuracy_hda:.4f}")

# Split data for O/U model
X_train_ou, X_test_ou, y_train_ou, y_test_ou = train_test_split(X, y_ou, test_size=0.2, random_state=42)

# Train O/U model
print("\nTraining O/U 2.5 model...")
ou_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
ou_model.fit(X_train_ou, y_train_ou)
y_pred_ou = ou_model.predict(X_test_ou)
accuracy_ou = accuracy_score(y_test_ou, y_pred_ou)
print(f"O/U 2.5 Model Accuracy: {accuracy_ou:.4f}")


# --- 4. Save Models ---
print("\n--- Saving Models ---")
joblib.dump(hda_model, 'hda_model.joblib')
joblib.dump(ou_model, 'ou_model.joblib')
print("Models saved to hda_model.joblib and ou_model.joblib")
