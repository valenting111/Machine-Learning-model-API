from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List, Dict
import uvicorn
import pickle
import pandas as pd


################ Creating the API ###############


class PredictRequest(BaseModel):
    features: Dict[str, float]
    

class ModelResponse(BaseModel):
    predictions: Optional[List[Dict[str, float]]]
    error: Optional[str]


app = FastAPI()

@app.get("/", response_model=ModelResponse)
async def root() -> ModelResponse:
    return ModelResponse(error="This is a test endpoint.")


@app.get("/predict", response_model=ModelResponse)
async def explain_api() -> ModelResponse:
    return ModelResponse(
        error="Send a POST request to this endpoint with 'features' data."
    )


@app.post("/predict")
async def get_model_predictions(request: PredictRequest) -> ModelResponse:   
    try:
        pickle_in = open("ensemble_classifier.pkl", "rb")
        classifier = pickle.load(pickle_in)
    except:
        print("Could not load model")
    try:
        inputs = request.features
        test_input = pd.DataFrame.from_records(inputs, index=[0])
    except:
        model_response = ModelResponse(error="Could not load all the input features properly")
        return model_response
        
    try:
        result = classifier.predict(test_input)[0]
        if result == 0:
            prediction = 0.0
        else:
            prediction = 1.0
    except Exception as e:
        print(e)
        model_response = ModelResponse(error="Could not successfully predict the result")
        return model_response
    
    
    return ModelResponse(predictions= [{"Prediction for death event": prediction}])
        

################ Training the model ###############


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB



def train_model(save_model=False):
    """
    

    This functions trains an ensemble model on the Heart failure clinical records Data Set
    and saves it optionally to disk.
    
    Returns None

    """
    
    
    #load the dataset
    df = pd.read_csv("heart_failure_clinical_records_dataset.csv")
    
    #Remove uncorrelated features and extract X and y columns
    df_removed_uncorr = df.drop(['diabetes', 'sex'], axis=1)
    X = df_removed_uncorr.drop(['DEATH_EVENT'], axis=1)
    y = df_removed_uncorr['DEATH_EVENT']
    
    
    #Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    
    #Feature preprocessing
    standard_scaler = StandardScaler() 
    one_hot = OneHotEncoder(handle_unknown='ignore')
    numerical_features = ["age", "creatinine_phosphokinase", "ejection_fraction", "platelets", "serum_creatinine", "serum_sodium", "time"]
    categorical_features = ["anaemia", "high_blood_pressure", "smoking"]
    
    transformer = ColumnTransformer(transformers=[("one_hot", one_hot, categorical_features),
                                     ("scaling", standard_scaler, numerical_features)], remainder="passthrough")
    
    
    #training the ensemble model
    ensemble_classifier = VotingClassifier(estimators=[ 
        ('SVC', SVC()),
        ('LogisticR', LogisticRegression()),
        ('RandomF', RandomForestClassifier())
        ], voting='hard')
    
    params = {'SVC__C': [0.1, 1., 10.],
              'SVC__kernel': ['linear', 'rbf'],
          'LogisticR__C': [0.1, 1., 10.],
             'RandomF__n_estimators': [20, 50, 100],
             'RandomF__max_features' : [2,4,6,8]}
    
    ensemble_gridsearch = GridSearchCV(estimator=ensemble_classifier, param_grid=params, cv=5)
    
    pipe = Pipeline([('transformer', transformer), ('ensemble_gridsearch', ensemble_gridsearch)])
    pipe.fit(X_train, y_train)
    
    if save_model:
        pickle_out = open("ensemble_classifier.pkl", "wb")
        pickle.dump(pipe, pickle_out)
        pickle_out.close()
        
    
    
################ Choosing which command to run ###############

import sys    

if __name__ == '__main__':
    for i, arg in enumerate(sys.argv):
            print(f"Argument {i:>6}: {arg}")
    if sys.argv[1] == "train":
        if sys.argv[2] == "1":
            train_model(True)
    elif sys.argv[1] == "api":
        uvicorn.run(app, host='127.0.0.1', port=8000)
    else:
        print("Command not recognized")
            
        
        
        
        
        
        
        
        
        
        
