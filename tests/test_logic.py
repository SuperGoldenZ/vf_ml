import pytest
import pandas as pd
import joblib
import vf_ml.data_helper
from sklearn.preprocessing import LabelEncoder


# Encode categorical data
def encode(data):
    label_encoder = LabelEncoder()
    data["Stage_encoded"] = label_encoder.fit_transform(data["Stage"])
    data["p1character_encoded"] = label_encoder.fit_transform(data["p1character"])
    data["p2character_encoded"] = label_encoder.fit_transform(data["p2character"])
    return data


test_data_accessing = [
    [
        "models/logistic_regression_demo.pkl",
        "Octagon",
        35,
        "Blaze",
        41,
        "Blaze",
        2,
        2,
        100,
        1,
        1,
        1,
        0.17,
    ],
    [
        "models/logistic_regression_demo.pkl",
        "Octagon",
        35,
        "Blaze",
        41,
        "Blaze",
        0,
        0,
        100,
        100,
        45,
        0.39,
        0.61,
    ],
    [
        "models/logistic_regression_demo.pkl",
        "River",
        21,
        "Akira",
        21,
        "Blaze",
        0,
        0,
        100,
        100,
        45,
        0.5,
        0.5,
    ],
    [
        "models/logistic_regression_demo.pkl",
        "Octagon",
        21,
        "Blaze",
        21,
        "Blaze",
        0,
        0,
        100,
        100,
        45,
        0.5,
        0.5,
    ],
    [
        "models/logistic_regression_demo.pkl",
        "Temple",
        21,
        "Akira",
        21,
        "Sarah",
        0,
        0,
        100,
        100,
        45,
        0.5,
        0.5,
    ],
    [
        "models/logistic_regression_demo.pkl",
        "Octagon",
        21,
        "Blaze",
        21,
        "Blaze",
        0,
        0,
        50,
        100,
        15,
        0.28,
        0.71,
    ],
    [
        "models/logistic_regression_demo.pkl",
        "Octagon",
        21,
        "Blaze",
        21,
        "Blaze",
        0,
        0,
        100,
        5,
        1,
        0.8,
        0.19,
    ],
    [
        "models/logistic_regression_demo.pkl",
        "River",
        21,
        "Blaze",
        21,
        "Blaze",
        0,
        0,
        100,
        5,
        1,
        0.8,
        0.19,
    ],
]


@pytest.mark.parametrize(
    "model_filename, stage, p1rank, p1character, p2rank, p2character,  p1_rounds_won_so_far, p2_rounds_won_so_far, p1health, p2health, time_remaining, expected_p1_wp, expected_p2_wp",
    test_data_accessing,
)
def test_data(
    model_filename,
    stage,
    p1rank,
    p1character,
    p2rank,
    p2character,
    p1_rounds_won_so_far,
    p2_rounds_won_so_far,
    p1health,
    p2health,
    time_remaining,
    expected_p1_wp,
    expected_p2_wp,
):
    p1_frame_data: vf_ml.PlayerFrameData = vf_ml.PlayerFrameData(
        rank=p1rank,
        health=p1health,
        rounds_won_so_far=p1_rounds_won_so_far,
        character=p1character,
        drinks=0,
        ringname="",
    )

    p2_frame_data: vf_ml.PlayerFrameData = vf_ml.PlayerFrameData(
        rank=p2rank,
        health=p2health,
        rounds_won_so_far=p2_rounds_won_so_far,
        character=p2character,
        drinks=0,
        ringname="",
    )

    frame_data: vf_ml.FrameData = vf_ml.FrameData(
        stage=stage,
        time_remaining=time_remaining,
        p1_frame_data=p1_frame_data,
        p2_frame_data=p2_frame_data,
    )

    new_data = vf_ml.DataHelper.create_data_frame(frame_data, version=2)

    model = joblib.load(model_filename)

    proba = model.predict_proba(new_data)  # Predict probabilities for both classes
    win_prob_player2 = proba[:, 1]  # Probability that Player 2 wins
    win_prob_player1 = proba[:, 0]  # Probability that Player 1 wins

    assert win_prob_player2 - expected_p2_wp < 0.1
    assert win_prob_player1 - expected_p1_wp < 0.1
