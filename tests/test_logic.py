import pytest
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Encode categorical data
def encode(data):
    label_encoder = LabelEncoder()
    data["Stage_encoded"] = label_encoder.fit_transform(data["Stage"])
    data["p1character_encoded"] = label_encoder.fit_transform(data["p1character"])
    data["p2character_encoded"] = label_encoder.fit_transform(data["p2character"])
    return data


test_data_accessing = [
    ["models/logistic_regression_demo.pkl", "Octagon", 35, "Blaze", 41, "Blaze", 100, 100, 45, 0.39, 0.61],
    ["models/logistic_regression_demo.pkl", "River", 21, "Akira", 21, "Blaze", 100, 100, 45, 0.5, 0.5],
    ["models/logistic_regression_demo.pkl", "Octagon", 21, "Blaze", 21, "Blaze", 100, 100, 45, 0.5, 0.5],
    ["models/logistic_regression_demo.pkl", "Temple", 21, "Akira", 21, "Sarah", 100, 100, 45, 0.5, 0.5],
    ["models/logistic_regression_demo.pkl", "Octagon", 21, "Blaze", 21, "Blaze", 50, 100, 15, 0.28, 0.71],
    ["models/logistic_regression_demo.pkl", "Octagon", 21, "Blaze", 21, "Blaze", 100, 5, 1, .8, .19],
    ["models/logistic_regression_demo.pkl", "River", 21, "Blaze", 21, "Blaze", 100, 5, 1, .8, .19],
    ]

@pytest.mark.parametrize(
    "model_filename, stage, p1rank, p1character, p2rank, p2character,  p1health, p2health, time_remaining, expected_p1_wp, expected_p2_wp", test_data_accessing
)
def test_data(model_filename, stage, p1rank, p1character, p2rank, p2character, p1health, p2health, time_remaining, expected_p1_wp, expected_p2_wp):
    new_data = pd.DataFrame(
        {
            "Player 1 Rank": [p1rank],
            "Player 2 Rank": [p2rank],
            "P1 Health": [p1health],
            "P2 Health": [p2health],
            "health_diff": [p1health - p2health],
            "Time Remaining When Round Ended": [time_remaining],
            "Stage": [stage],
            "p1character": [p1character],
            "p2character": [p2character],
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
    print(f"{win_prob_player1} {win_prob_player2}")
    assert(win_prob_player2 - expected_p2_wp < .1)
    assert(win_prob_player1 - expected_p1_wp < .1)
