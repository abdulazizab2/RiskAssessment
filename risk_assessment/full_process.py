import training
import scoring
import deployment
import diagnostics
import reporting
import json
import os
import subprocess
import sys
import pickle
import pandas as pd
from risk_assessment.common import preprocess_data
from risk_assessment.scoring import score_model
from risk_assessment.utils.logger import logging


def get_ingested_files(deployment_path):
    with open(deployment_path, "r") as f:
        ingested_files = f.read().splitlines()
    return ingested_files


def check_new_files(ingested_files, source_data_path):
    for file in os.listdir(source_data_path):
        if not file.endswith(".csv"):
            continue
        if file not in ingested_files:
            return True
    return False


def check_model_drift(deployment_path, final_data_path):
    with open(os.path.join(deployment_path, "latest_score.txt"), "r") as f:
        score = float(f.read())
    with open(os.path.join(deployment_path, "trained_model.pkl"), "rb") as f:
        model = pickle.load(f)
    data = pd.read_csv(final_data_path)
    X, y = preprocess_data(data)
    new_score = score_model(
        model=model, data=(X, y), save_result=False, results_path=None
    )
    return new_score > score


def main():
    with open("config.json", "r") as f:
        config = json.load(f)
    deployment_path = config["prod_deployment_path"]
    ingested_files_path = os.path.join(
        config["prod_deployment_path"], "ingested_files.txt"
    )
    source_data_path = config["input_folder_path"]
    ingested_files = get_ingested_files(ingested_files_path)
    is_new_data = check_new_files(ingested_files, source_data_path)
    if not is_new_data:
        logging.info("No recent data ingested. Exiting")
        sys.exit(-1)
    subprocess.Popen(["python", "risk_assessment/ingestion.py"])
    logging.info("SUCCESS: New data ingested")
    final_data_path = os.path.join(config["output_folder_path"], "final_data.csv")
    is_drifted = check_model_drift(deployment_path, final_data_path)
    if not is_drifted:
        logging.info("Model has not drifted, redeploying terminated. Exiting")
        sys.exit(-1)
    subprocess.Popen(["python", "risk_assessment/training.py"])
    logging.info("SUCCESS: New model trained")
    subprocess.Popen(["python", "risk_assessment/deployment.py"])
    logging.info("SUCCESS: New trained model is deployed")
    subprocess.Popen(["uvicorn", "app:app"])
    subprocess.Popen(["python", "risk_assessment/api_calls.py"])
    subprocess.Popen(["python", "risk_assessment/reporting.py"])
    logging.info("SUCCESS: Results are saved in models/ directory")


if __name__ == "__main__":
    main()
