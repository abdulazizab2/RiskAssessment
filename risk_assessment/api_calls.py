import requests
import json
import os


def main():
    URL = "http://127.0.0.1:8000"
    with open("config.json", "r") as f:
        config = json.load(f)
    model_path = os.path.join(config["prod_deployment_path"], "trained_model.pkl")
    data_path = os.path.join(config["test_data_path"], "testdata.csv")
    responses_output_path = os.path.join(config["output_model_path"])
    data = {"data_path": data_path}
    response1 = requests.post(f"{URL}/prediction", params=data)
    response2 = requests.post(f"{URL}/scoring")
    response3 = requests.post(f"{URL}/summarystats")
    response4 = requests.post(f"{URL}/diagnostics")

    responses = [response1.text, response2.text, response3.text, response4.text]
    with open(os.path.join(responses_output_path, "api_returns.txt"), "w") as f:
        for response in responses:
            f.write(response + "\n")


if __name__ == "__main__":
    main()
