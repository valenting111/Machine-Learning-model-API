# Machine-Learning-model-API


This project trains a machine learning model on the Heart failure clinical records Data Set available at https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records , and implements an API to use that model in a (local) web server for prediction.


#### What the repository contains:

* EnsembleModel.ipynb : An Jupyter Notebook that performs the analysis and visualization of the dataset, trains an ensemble model on it and evaluate the results.
* heart_failure_clinical_records_dataset.csv : The dataset in question.
* api.py : The main script. It allows to run different actions based on the user's input commands. 
* test_data.json : A JSON file containing a test input sample .
* references.txt : A list of the resources used to achieve this project.
* ensemble_classifier.pkl : The trained model in pickle format. It is possible to overwrite it by training a new model through api.py.
* requirements.txt : A list of the required packages to be able to run all the scripts.



For this project, I created a new virtual environment using Anaconda, the Spyder IDE and Python 3.8.5.
The list of the packages required (liste in "requirements.txt") are:
* pandas
* numpy
* fastapi
* pydantic
* uvicorn
* sklearn
* pickle
* sys
* typing


The Jupyter Notebook is simply used to explain visually and analytically the motivations behind certain choices, like preprocessing or removing some input features. But everything can be executed through the "api.py" file.


#### How to use api.py

This script can be ran for 2 different purposes:
* Train an ensemble model on the dataset mentioned above and optionally save it.
* Start a server on your local machine to use the trained model through an API.

In both cases, open your terminal or anaconda prompt and navigate to the folder where this script is located.
Then, to choose with part of the script to run, the python file reads in the command line arguments you feed it with.
So to train a model without saving it run: `python api.py train 0`


To do the same but saving the model run: `python api.py train 1`


To start the server on your local machine, run: `python api.py api`


Once the server is started, you will be able to make predictions on new test data.

##### Making predictions with curl

The server will be hosted at: http://127.0.0.1:8000
So to make a POST request to it and ask for a prediction on the test sample in test_data.json, you need to open a new terminal, navigate to the same folder where these files are located, and run: `curl --request POST --data @test_data.json http://127.0.0.1:8000/predict`


#### Note:
* While the server URL is fixed, the JSON file can be renamed.
* As explained in the Jupyter Notebook, the 'diabetes' and 'sex' features are not used, hence they should not be present in the JSON file.

















