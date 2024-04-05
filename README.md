# Depression-Prediction-Using-BiLSTM

## Dataset
The dataset used for this project is sourced from Kaggle. You can access the dataset [here](https://www.kaggle.com/datasets/infamouscoder/depression-reddit-cleaned).

## Overview
This project aims to predict depression using a Bidirectional Long Short-Term Memory (BiLSTM) model trained on the given dataset. The dataset consists of cleaned Reddit posts related to depression, with labels indicating whether each post is associated with depression or not.

## Code Description
The provided code implements the following steps:

1. Data Loading: Reads the dataset "depression_dataset_reddit_cleaned.csv" into a Pandas DataFrame.
2. Data Preprocessing: Performs data cleaning and preprocessing on the text data, including tokenization, lemmatization, and removing stopwords.
3. Model Building: Constructs a BiLSTM model using TensorFlow's Keras API, with an embedding layer, Bidirectional LSTM layer, and a dense output layer with sigmoid activation.
4. Model Training: Trains the BiLSTM model on the preprocessed text data, splitting the dataset into training and testing sets.
5. Model Evaluation: Evaluates the trained model on the testing set using accuracy, precision, recall, F1-score, and confusion matrix.

## Requirements
- Python 3.x
- TensorFlow
- Pandas
- Matplotlib
- Seaborn
- NLTK

## Usage
1. Clone this repository to your local machine.
2. Download the dataset from the provided Kaggle link and save it as "depression_dataset_reddit_cleaned.csv" in the repository folder.
3. Install the required Python packages specified in the `requirements.txt` file.
4. Run the provided Python script.

## Results
The model performance metrics, including accuracy, precision, recall, F1-score, and confusion matrix, are printed to the console after model training and evaluation.
