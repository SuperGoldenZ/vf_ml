import pandas as pd
import argparse
import joblib

import vf_ml.data_helper

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, brier_score_loss

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
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Random Forest model
print("Creating RandomForestClassifier")
model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    #max_depth=10,  # Limit how deep trees can grow
    #min_samples_split=10,  # Minimum samples to split a node
    #min_samples_leaf=5,  # Minimum samples at a leaf node
    #max_features="sqrt",  # Random subset of features
)

print("Fitting")
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(
    classification_report(
        y_test, y_pred, target_names=["Player 1 Wins", "Player 2 Wins"]
    )
)

brier = brier_score_loss(y_test, y_proba[:, 1])  # Lower is better
print(f"Brier Score Loss: {brier:.4f}")

feature_importance = pd.DataFrame(
    {"Feature": X.columns, "Importance": model.feature_importances_}
).sort_values(by="Importance", ascending=False)

print(feature_importance)

# Save the model
if predict == "Match Winner":
    joblib.dump(model, "models/random_forest_match_winner.pkl")
else:
    joblib.dump(model, "models/random_forest_round_winner.pkl")
