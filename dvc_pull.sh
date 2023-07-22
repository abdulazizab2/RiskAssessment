#! /bin/bash

dvc fetch source_data/dataset4.csv.dvc
dvc pull source_data/dataset4.csv.dvc
dvc fetch source_data/dataset3.csv.dvc
dvc pull source_data/dataset3.csv.dvc
dvc fetch ingested_data/ingested_files.txt.dvc
dvc pull ingested_data/ingested_files.txt.dvc
dvc fetch ingested_data/final_data.csv.dvc
dvc pull ingested_data/final_data.csv.dvc
dvc fetch practice_data/dataset1.csv.dvc
dvc pull practice_data/dataset1.csv.dvc
dvc fetch practice_data/dataset2.csv.dvc
dvc pull practice_data/dataset2.csv.dvc
dvc fetch practice_models/latest_score.txt.dvc
dvc pull practice_models/latest_score.txt.dvc
dvc fetch practice_models/trained_model.pkl.dvc
dvc pull practice_models/trained_model.pkl.dvc
dvc fetch test_data/testdata.csv.dvc
dvc pull test_data/testdata.csv.dvc
dvc fetch models/latest_score.txt.dvc
dvc pull models/latest_score.txt.dvc
dvc fetch models/trained_model.pkl.dvc
dvc pull models/trained_model.pkl.dvc
