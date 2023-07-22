from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn
import pandas as pd
import numpy as np
import pickle
from typing import Optional
import json
import os
from risk_assessment.utils.logger import logging
from risk_assessment.diagnostics import (
    model_predictions,
    dataframe_summary,
    report_missing_data,
    execution_time,
    outdated_packages_list,
)
from risk_assessment.scoring import score_model
from risk_assessment.common import preprocess_data

app = FastAPI()


with open("config.json", "r") as f:
    config = json.load(f)
model_path = os.path.join(config["prod_deployment_path"], "trained_model.pkl")
data_path = os.path.join(config["test_data_path"], "testdata.csv")


@app.post("/prediction")
def predict(data_path: str):
    preds, _ = model_predictions(model_path, data_path)
    return json.dumps(preds.tolist())


@app.post("/scoring")
def stats(data_path: Optional[str] = data_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    X, y = preprocess_data(pd.read_csv(data_path))
    f1_score_ = score_model(model, data=(X, y), save_result=False)
    return json.dumps(f1_score_)


@app.post("/summarystats")
def stats(data_path: Optional[str] = data_path):
    stats = dataframe_summary(data_path)
    return json.dumps(stats)


@app.post("/diagnostics")
def stats(data_path: Optional[str] = data_path):
    missing_data = report_missing_data(data_path)
    time_data = execution_time()
    dependency_data = outdated_packages_list()
    return {
        "missing_data": missing_data,
        "timing_stats": time_data,
        "dependency_report": dependency_data,
    }


if __name__ == "__main__":
    uvicorn.run(host="0.0.0.0", port=8000)
