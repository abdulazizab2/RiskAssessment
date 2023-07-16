import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LogisticRegression
import json
from risk_assessment.utils.logger import logging
from risk_assessment.common import preprocess_data


def train_model(X, y, model_output_path):
    lr = LogisticRegression(
        C=1.0,
        class_weight=None,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        l1_ratio=None,
        max_iter=100,
        multi_class="auto",
        n_jobs=None,
        penalty="l2",
        random_state=0,
        solver="liblinear",
        tol=0.0001,
        verbose=0,
        warm_start=False,
    )

    model = lr.fit(X, y)
    if not os.path.exists(model_output_path):
        os.mkdir(model_output_path)
    with open(os.path.join(model_output_path, "trained_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    logging.info(f"SUCCESS: Model is trained saved in {model_output_path}/ folder")


def main():
    with open("config.json", "r") as f:
        config = json.load(f)

    dataset_csv_path = os.path.join(config["output_folder_path"])
    model_path = os.path.join(config["output_model_path"])
    data = pd.read_csv(os.path.join(dataset_csv_path, "final_data.csv"))
    X, y = preprocess_data(data)
    train_model(X, y, model_path)


if __name__ == "__main__":
    main()
