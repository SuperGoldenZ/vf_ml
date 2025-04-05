import pandas as pd
import argparse
import joblib

import vf_ml.data_helper

from sklearn.model_selection import train_test_split
#from cuml.linear_model import LogisticRegression
#from dask_ml.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
#from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, brier_score_loss
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser(description="Train from match data file")
parser.add_argument("--filename", help="csv or gzip file", type=str, required=True)
parser.add_argument(
    "--predict",
    help="Match Winner or Round Winner",
    choices=["Match Winner", "Round Winner"],
    type=str,
    required=True,
)

# Parse arguments
args = parser.parse_args()

# Access the filename
filename = args.filename
predict = args.predict

if predict == "Round Winner":
    predict = "Winning Player Number"

print(f"Processing file: {filename} to predict {predict}")

data = vf_ml.DataHelper.load_data(filename)

# Define features and target
X = vf_ml.data_helper.DataHelper.get_x_features(data, 2)

y = (data[predict] == 2).astype(int)  # Player 2 win: 1, Player 1 win: 0

print(y.value_counts())

print("Splitting data")
# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Logistic regression model
print("Creating LogicsticRegression")
model = LogisticRegression(max_iter=2500, class_weight="balanced", random_state=42)


print("Fitting")
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)  # Probabilities

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(
    classification_report(
        y_test, y_pred, target_names=["Player 1 Wins", "Player 2 Wins"]
    )
)

# Brier score for probability calibration
brier = brier_score_loss(y_test, y_proba[:, 1])  # Lower is better
print(f"Brier Score Loss: {brier:.4f}")

feature_importance = pd.DataFrame(
    {"Feature": X.columns, "Weight": model.coef_[0]}
).sort_values(by="Weight", ascending=False)

print(feature_importance)

if predict == "Match Winner":
    joblib.dump(model, "models/logistic_regression_match_winner.pkl")
else:
    joblib.dump(model, "models/logistic_regression_round_winner.pkl")
