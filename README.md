# ALTEGRAD Challenge Fall 2018

## Contact information
Kaggle in-class team name: AitAzzi - Ohleyer - Sutton

For any question/request related to this code, please send an email to on of these addresses: abderrahim.aitazzi@ens-paris-saclay.fr, sebastien.ohleyer@gmail.com, michael.sutton@student.ecp.fr.


# Code organization

## Feature generation
Each feature can be generated from the `Feature_generation.ipynb` notebook. This notebook query `.py` files in the directory feature_engineering.

Each `.py` computes their corresponding features and write them in a CSV file in the data directory. For simplicity, we directly provide CSV files containing every features. 

## LightGB
Use the `Lightgb_classification.ipynb` notebook to perform classification. Our parameters leading to our best submission are provided in this notebook (a random initialization of the model set apart).

This notebook calls the function `load_features.py` which load features from every CSV files in the data directory and combine them in a unique pandas DataFrame.

For each classifier fitting, a CSV file containing, the model, its parameters and cross-validation results will be written in the log directory. To compare results with our previous fittings we also provide some of our results.

## Neural networks


# Requirements
Every package used in these codes were install with their latest version, except for:
- NetworkX: v1.11

