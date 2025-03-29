import pytest
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, brier_score_loss

# Encode categorical data
def encode(data):
    label_encoder = LabelEncoder()
    data["Stage_encoded"] = label_encoder.fit_transform(data["Stage"])
    data["p1character_encoded"] = label_encoder.fit_transform(data["p1character"])
    data["p2character_encoded"] = label_encoder.fit_transform(data["p2character"])
    return data


test_data_accessing = [["models/logistic_regression_demo.pkl", 100, 100, 45]]


@pytest.mark.parametrize(
    "model_filename,p1health, p2health, time_remaining", test_data_accessing
)
def test_data(model_filename, p1health, p2health, time_remaining):
    new_data = pd.DataFrame(
        {
            "P1 Health": [p1health],
            "P2 Health": [p2health],
            "health_diff": [p1health - p2health],
            "Time Remaining When Round Ended": [time_remaining],
            "Stage": ["Octagon"],
            "p1character": ["Blaze"],
            "p2character": ["Blaze"],
        }
    )

    new_data = encode(new_data)
    del new_data["p1character"]
    del new_data["p2character"]
    del new_data["p1character_encoded"]
    del new_data["p2character_encoded"]
    del new_data["Stage"]

    new_data["health_time_interaction"] = [
        (p1health - p2health) * (45 - time_remaining)
    ]

    model = joblib.load(model_filename)
    proba = model.predict_proba(new_data)  # Predict probabilities for both classes
    win_prob_player2 = proba[:, 1]  # Probability that Player 2 wins
    win_prob_player1 = proba[:, 0]  # Probability that Player 1 wins

    print(f"\n{p1health} vs {p2health} - {time_remaining} seconds")
    print(
        f"Win probabilities p1 {win_prob_player1[:10]} p2 {win_prob_player2[:10]}",
    )
