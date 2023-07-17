import pandas as pd
import numpy as np
import os
import shutil
import json
from risk_assessment.utils.logger import logging


def copy_files_to_deployment_directory(files, deployment_path):
    if not os.path.exists(deployment_path):
        os.mkdir(deployment_path)
    for file in files:
        shutil.copy(file, deployment_path)
    logging.info("SUCCESS: Model is deployed")


def main():
    with open("config.json", "r") as f:
        config = json.load(f)

    dataset_csv_path = os.path.join(config["output_folder_path"])
    model_path = os.path.join(config["output_model_path"])
    prod_deployment_path = os.path.join(config["prod_deployment_path"])
    files = []
    files.append(os.path.join(dataset_csv_path, "ingested_files.txt"))
    files.append(os.path.join(model_path, "trained_model.pkl"))
    files.append(os.path.join(model_path, "latest_score.txt"))
    copy_files_to_deployment_directory(files, deployment_path=prod_deployment_path)


if __name__ == "__main__":
    main()
