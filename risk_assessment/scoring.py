import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import f1_score
import json
from risk_assessment.utils.logger import logging
from risk_assessment.common import preprocess_data


def score_model(model, data, results_path=None, save_result=True):
    X, y = data
    preds = model.predict(X)
    f1_score_ = f1_score(y, preds)
    if save_result:
        with open(os.path.join(results_path, "latest_score.txt"), "w") as f:
            f.write(f"{f1_score_:0.4f}")
        logging.info(f"SUCCESS: Model score is saved in {results_path}/ directory")
    return f1_score_


def main():
    with open("config.json", "r") as f:
        config = json.load(f)

    model_path = os.path.join(config["output_model_path"])
    test_data_path = os.path.join(config["test_data_path"])
    with open(os.path.join(model_path, "trained_model.pkl"), "rb") as f:
        model = pickle.load(f)
    test_data = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))
    X, y = preprocess_data(test_data)
    score_model(model, data=(X, y), results_path=model_path)


if __name__ == "__main__":
    main()
