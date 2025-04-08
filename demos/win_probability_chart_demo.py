import os

import vf_ml.data_helper
import argparse

parser = argparse.ArgumentParser(description="File with CSV data from a match")
parser.add_argument("--filename", help="csv or gzip file", type=str, required=True)
parser.add_argument("--model_filename", help="pkl file", type=str, required=True)

args = parser.parse_args()

# Access the filename
filename = args.filename

if not os.path.exists(filename):
    print(f"{filename} does not exist")
    exit(1)

model_filename = args.model_filename
if not os.path.exists(model_filename):
    print(f"{model_filename} does not exist")
    exit(1)

data = vf_ml.DataHelper.load_data(filename, prepare_for_prediction=True)
win_probability_chart = vf_ml.WinProbabilityChart(match_model_filename=model_filename)

win_probability_chart.generate_match_win_prob_chart_with_single_line(
    frame_data=data,
    p1character=data["Player 1 Character"].iloc[0],
    p2character=data["Player 2 Character"].iloc[0],    
)
