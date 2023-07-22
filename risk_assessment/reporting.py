import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from risk_assessment.common import preprocess_data
from risk_assessment.diagnostics import model_predictions


def score_model(y_true, y_pred, figure_output_path):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.savefig(os.path.join(figure_output_path, "confusion_matrix.png"))


if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)
    test_data_path = os.path.join(config["test_data_path"], "testdata.csv")
    model_path = os.path.join(config["prod_deployment_path"], "trained_model.pkl")
    figure_output_path = os.path.join(config["output_model_path"])
    preds, y = model_predictions(model_path, test_data_path)
    score_model(y_true=y, y_pred=preds, figure_output_path=figure_output_path)
