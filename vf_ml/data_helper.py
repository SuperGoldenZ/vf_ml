"""Class for processing VF data for machine learning"""

import pandas as pd
from line_profiler import profile
from vf_ml import frame_data
from sklearn.preprocessing import LabelEncoder


class DataHelper:
    """Methods for processing VF data for machine learning"""

    @staticmethod
    def encode(data, version=2):
        """Encode categorical data"""
        label_encoder = LabelEncoder()
        data["Stage_encoded"] = label_encoder.fit_transform(data["Stage"])

        return data

    @staticmethod
    def load_data(filename="vf_match_data.csv", prepare_for_prediction=False):
        """Load and pre-process the data"""

        if ".gz" in filename:
            data = pd.read_csv(filename, compression="gzip")
        elif ".csv" in filename:
            data = pd.read_csv(filename)

        data = data[data["Time Remaining When Round Ended"] != "NA"]
        data = data.dropna(subset=["P1 Health"])
        data = data.dropna(subset=["P2 Health"])
        data = data.dropna(subset=["Time Remaining When Round Ended"])
        data = data.dropna(subset=["Player 1 Rank"])
        data = data.dropna(subset=["Player 2 Rank"])
        data = data.dropna(subset=["Shun.Drinks.1P"])
        data = data.dropna(subset=["Shun.Drinks.2P"])

        data["health_diff"] = data["P1 Health"] - data["P2 Health"]
        data["rank_diff"] = data["Player 1 Rank"] - data["Player 2 Rank"]

        if data["Time Remaining When Round Ended"] is None:
            data["Time Remaining When Round Ended"] = 45

        data["health_time_interaction"] = data["health_diff"] * (
            45 - data["Time Remaining When Round Ended"].astype(int)
        )
        data["p1_drinks_time"] = data["Shun.Drinks.1P"] * (
            data["Time Remaining When Round Ended"].astype(int)
        )
        data["p2_drinks_time"] = data["Shun.Drinks.2P"] * (
            data["Time Remaining When Round Ended"].astype(int)
        )

        if prepare_for_prediction:
            data = DataHelper.create_data_frame(frame_data=None, version=2, df=data)
        else:
            data = DataHelper.encode(data)
        return data

    @staticmethod
    def get_x_features(data, version=1):
        """Retruns x features of the dataset"""
        if version == 1:
            return data[
                [
                    "P1 Health",
                    "P2 Health",
                    "health_diff",
                    "Time Remaining When Round Ended",
                    "Player 1 Rank",
                    "Player 2 Rank",
                    "rank_diff",
                    "P1 Rounds Won So Far",
                    "P2 Rounds Won So Far",
                    "Stage_encoded",
                    "health_time_interaction",
                ]
            ]
        if version == 2:
            return data[
                [
                    "P1 Health",
                    "P2 Health",
                    "health_diff",
                    # "Time Remaining When Round Ended",
                    # "Player 1 Rank",
                    # "Player 2 Rank",
                    "rank_diff",
                    "P1 Rounds Won So Far",
                    "P2 Rounds Won So Far",
                    "Shun.Drinks.1P",
                    "Shun.Drinks.2P",
                    "Stage_encoded",
                    # "Player 1 Ringname Encoded",
                    # "Player 2 Ringname Encoded",
                    "health_time_interaction",
                    "p1_drinks_time",
                    "p2_drinks_time",
                ]
            ]

    @staticmethod
    def get_linear_features():
        """Return linear features"""
        return [
            "P1 Health",
            "P2 Health",
            "health_diff",
            "Time Remaining When Round Ended",
            "rank_diff",
            "health_time_interaction",
        ]

    @staticmethod
    def get_nonlinear_features():
        """Return nonlinear features"""
        return [
            "Player 1 Rank",
            "Player 2 Rank",
            "P1 Rounds Won So Far",
            "P2 Rounds Won So Far",
        ]

    @staticmethod
    @profile
    def create_data_frame(
        frame_data: frame_data.FrameData,
        version=1,
        df: pd.DataFrame = None,
    ):
        """For creating a test frame"""
        if version == 1 and df is None:
            new_data = pd.DataFrame(
                {
                    "P1 Health": [frame_data.p1_frame_data.health],
                    "P2 Health": [frame_data.p2_frame_data.health],
                    "health_diff": [
                        frame_data.p1_frame_data.health
                        - frame_data.p2_frame_data.health
                    ],
                    "Time Remaining When Round Ended": [frame_data.time_remaining],
                    "Stage": [frame_data.stage],
                    "Player 1 Rank": [frame_data.p1_frame_data.rank],
                    "Player 2 Rank": [frame_data.p2_frame_data.rank],
                    "rank_diff": [
                        frame_data.p1_frame_data.rank - frame_data.p2_frame_data.rank
                    ],
                    "P1 Rounds Won So Far": [
                        frame_data.p1_frame_data.rounds_won_so_far
                    ],
                    "P2 Rounds Won So Far": [
                        frame_data.p2_frame_data.rounds_won_so_far
                    ],
                }
            )
        elif version == 2 and df is None:
            new_data = pd.DataFrame(
                {
                    "P1 Health": [frame_data.p1_frame_data.health],
                    "P2 Health": [frame_data.p2_frame_data.health],
                    "health_diff": [
                        frame_data.p1_frame_data.health
                        - frame_data.p2_frame_data.health
                    ],
                    "Time Remaining When Round Ended": [frame_data.time_remaining],
                    "Stage": [frame_data.stage],
                    "Player 1 Rank": [frame_data.p1_frame_data.rank],
                    "Player 2 Rank": [frame_data.p2_frame_data.rank],
                    "rank_diff": [
                        frame_data.p1_frame_data.rank - frame_data.p2_frame_data.rank
                    ],
                    "P1 Rounds Won So Far": [
                        frame_data.p1_frame_data.rounds_won_so_far
                    ],
                    "P2 Rounds Won So Far": [
                        frame_data.p2_frame_data.rounds_won_so_far
                    ],
                    "Shun.Drinks.1P": [frame_data.p1_frame_data.drinks],
                    "Shun.Drinks.2P": [frame_data.p2_frame_data.drinks],
                    "Player 1 Ringname": [frame_data.p1_frame_data.ringname],
                    "Player 2 Ringname": [frame_data.p2_frame_data.ringname],
                }
            )

        if df is not None:
            new_data = df

        new_data = DataHelper.encode(new_data, version)
        del new_data["Stage"]

        if version == 2:
            del new_data["Player 1 Ringname"]
            del new_data["Player 2 Ringname"]

        if frame_data is not None:
            new_data["health_time_interaction"] = [
                new_data["health_diff"] * (45 - int(frame_data.time_remaining))
            ]

        if version == 2:
            new_data["p1_drinks_time"] = new_data["Shun.Drinks.1P"] * (
                new_data["Time Remaining When Round Ended"].astype(int)
            )
            new_data["p2_drinks_time"] = new_data["Shun.Drinks.2P"] * (
                new_data["Time Remaining When Round Ended"].astype(int)
            )

        if version == 2 and df is None:
            del new_data["Player 1 Rank"]
            del new_data["Player 2 Rank"]
            del new_data["Time Remaining When Round Ended"]

        return new_data
