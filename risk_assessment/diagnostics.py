import pandas as pd
import numpy as np
import timeit
import os
import subprocess
import requests
import json
import pickle
from risk_assessment.common import preprocess_data
from risk_assessment.utils.logger import logging


def model_predictions(model_path, test_data_path):
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except Exception as err:
        logging.error(f"{err}. Make sure you have deployed the model successfully")
        raise
    data = pd.read_csv(test_data_path)
    X, y = preprocess_data(data)
    preds = model.predict(X)
    logging.info("SUCCESS: Deployed model made predictions against test data")
    return preds, y


def dataframe_summary(data_path):
    data = pd.read_csv(data_path)
    data.drop(columns=["corporation"], inplace=True)
    summary_list = []
    for column in data.columns:
        column_dict = {
            "Column": column,
            "Mean": data[column].mean(),
            "Median": data[column].median(),
            "Std Dev": data[column].std(),
        }
        summary_list.append(column_dict)
    return summary_list


def report_missing_data(data_path):
    data = pd.read_csv(data_path)
    missing_data_dict = {}
    for column in data.columns:
        na_count = data[column].isna().sum()
        na_percent = (na_count / len(data)) * 100
        missing_data_dict[column] = na_percent
    return missing_data_dict


def execution_time():
    timing_dict = {"training_time": [], "data_ingestion_time": []}
    for _ in range(10):
        start_time = timeit.default_timer()
        subprocess.Popen(["python", "risk_assessment/training.py"])
        timing_dict["training_time"].append(timeit.default_timer() - start_time)
        start_time = timeit.default_timer()
        subprocess.Popen(["python", "risk_assessment/ingestion.py"])
        timing_dict["data_ingestion_time"].append(timeit.default_timer() - start_time)

    return timing_dict


def outdated_packages_list():
    table = pd.DataFrame(columns=["package_name", "current_version", "latest_version"])
    with open("requirements.txt", "r") as f:
        packages_list = f.read().split()
        for package in packages_list:
            name, version = package.split("==")[0], package.split("==")[1]
            r = requests.get(f"https://pypi.org/pypi/{name}/json")
            try:
                latest_version = r.json()["info"]["version"]
                table.loc[-1] = {
                    "package_name": name,
                    "current_version": version,
                    "latest_version": latest_version,
                }
                table.index = table.index + 1
                table.sort_index(inplace=True)
            except KeyError:
                logging.warn(f"DIAGNOSTICS: package: {name} not found in pypi")
    return table


if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)

    dataset_csv_path = os.path.join(config["output_folder_path"], "final_data.csv")
    test_data_path = os.path.join(config["test_data_path"], "testdata.csv")
    model_path = os.path.join(config["prod_deployment_path"], "trained_model.pkl")
    preds = model_predictions(model_path, test_data_path)
    summary = dataframe_summary(dataset_csv_path)
    print(f"-------\nData Summary:\n{summary}\n-------")
    missing_data = report_missing_data(dataset_csv_path)
    print(f"-------\nMissing Data:\n{missing_data}\n-------")
    timing = execution_time()
    print(f"-------\nExecution Timings:\n{timing}\n-------")
    dependency_table = outdated_packages_list()
    print(f"-------\nDependency Table:\n{dependency_table}\n-------")
