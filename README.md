# ALTEGRAD Challenge Fall 2018

This repo is about the Kaggle in-class challenge https://www.kaggle.com/c/altegrad-challenge-fall-17

This challenge was inspired by the Quora dataset challenge https://www.kaggle.com/c/quora-question-pairs

We finish at the #1 place on the Private and Public leaderboard.

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
Use the `Neural_net.ipynb` notebook to perform classification using the three networks we experimented (Fully connected, LSTM, LSTM with features). The notebook is divided in three sections (you should always run the first section).
1. Load data and features
2. Fully connected layer  
   This part is divided in the following subpart spliting the data into training and validation sets, model definition, training, loss visualisation, test prediction.  

3. LSTM  
   We first prepare the sentences to feed the neural network then split the data into training and validation sets. After this preparation step, we have two similiar pieces of code corresponding respectively to the network that uses only text and the one that uses also features. They have both the same structure as describe in 2. .  
   In the model definition you can comment or uncomment lines to choose weither to use multiplication or concatenation merging method and weither to use L2 normalisation or not.

# Requirements
Every package used in these codes were install with their latest version, except for:
- NetworkX: v1.11

