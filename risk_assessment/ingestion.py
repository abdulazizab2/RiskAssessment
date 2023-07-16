import subprocess
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from risk_assessment.utils.logger import logging


def merge_multiple_dataframe(config):
    input_folder_path = config["input_folder_path"]
    output_folder_path = config["output_folder_path"]
    df = pd.DataFrame()
    ingested_files = ""
    for file in os.listdir(input_folder_path):
        if file.endswith(".csv"):
            current_df = pd.read_csv(os.path.join(input_folder_path, file))
            df = pd.concat([df, current_df])
            ingested_files += file + "\n"
    df.drop_duplicates(inplace=True)
    if not os.path.exists(output_folder_path):
        os.mkdir(output_folder_path)
    df.to_csv(os.path.join(output_folder_path, "final_data.csv"), index=False)
    with open(os.path.join(output_folder_path, "ingested_files.txt"), "w") as f:
        f.write(ingested_files)
    logging.info(
        f"SUCCESS: Source data has been ingested and saved to {output_folder_path}/ folder"
    )


if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)
    merge_multiple_dataframe(config)
