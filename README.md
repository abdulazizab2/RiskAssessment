# Introduction
Machine learning repository for assessing attrition risks of clients. The repository aims to emphasize an end2end process of the machine learning lifecycle. Including usage of script automation, cronjobs and Data Version Control

# Links
[GitHub](https://github.com/abdulazizab2/RiskAssessment)

# Usage
Install requirements: ```pip install .```
Optional steps:
```bash
python risk_assessment/ingestion.py
python risk_assessment/training.py
python risk_assessment/scoring.py
python risk_assessment/deployment.py
python risk_assessment/diagnostics.py
uvicorn app:app
python risk_assessment/api_calls.py
```
You may ignore the optional steps by pulling the models and data using ```dvc pull```
**Note**: If dvc pulls fails. You may run ```sh dvc_pull.sh``` as a workaround solution

Finally, you can initialize the full process after you pull the artifacts from DVC by:
```bash
python risk_assessment/full_process.py
```