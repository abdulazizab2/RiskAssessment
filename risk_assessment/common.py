import pandas as pd


def preprocess_data(data: pd.DataFrame):
    X = data.drop(columns=["corporation"])
    y = data["exited"]
    return X, y
