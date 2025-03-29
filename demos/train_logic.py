import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, brier_score_loss

import argparse
parser = argparse.ArgumentParser(description="Train from match data file")
parser.add_argument("--filename", help="csv or gzip file", type=str)

# Parse arguments
args = parser.parse_args()

# Access the filename
filename = args.filename
print(f"Processing file: {filename}")

# Encode categorical data
def encode(data):
    label_encoder = LabelEncoder()
    data['Stage_encoded'] = label_encoder.fit_transform(data['Stage'])
    data['p1character_encoded'] = label_encoder.fit_transform(data['Player 1 Character'])
    data['p2character_encoded'] = label_encoder.fit_transform(data['Player 2 Character'])
    return data

# Load and preprocess the data
data = pd.read_csv(filename)
data = data[data['Time Remaining When Round Ended'] != "NA"]
data = data.dropna(subset=['P1 Health'])
data = data.dropna(subset=['Player 1 Rank'])
data = data.dropna(subset=['Player 2 Rank'])
data = data.dropna(subset=['Time Remaining When Round Ended'])
data = data.dropna(subset=['Shun.Drinks.1P'])

data['health_diff'] = data['P1 Health'] - data['P2 Health']
data['health_time_interaction'] = data['health_diff'] * (45 - data['Time Remaining When Round Ended'])
data = encode(data)

# Define features and target
X = data[['Player 1 Rank', 'Player 2 Rank', 'P1 Health', 'P2 Health', 'health_diff', 'Time Remaining When Round Ended', 'Stage_encoded', 'health_time_interaction']]
#print(X[X.isnull().any(axis=1)])

y = (data['Match Winner'] == 2).astype(int)  # Player 2 win: 1, Player 1 win: 0

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic regression model
model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
#model = LogisticRegression(max_iter=1000,  random_state=42)
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)  # Probabilities

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Player 1 Wins', 'Player 2 Wins']))

# Brier score for probability calibration
brier = brier_score_loss(y_test, y_proba[:, 1])  # Lower is better
print(f"Brier Score Loss: {brier:.4f}")

# Save the model
import joblib
joblib.dump(model, 'models/logistic_regression_demo.pkl')