from sklearn.model_selection import train_test_split, GridSearchCV
import yaml
import pandas as pd
import pickle
import os
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from mlflow.models import infer_signature
from urllib.parse import urlparse
from sklearn.metrics import confusion_matrix

os.environ['MLFLOW_TRACKING_URI']="http://127.0.0.1:5000"

def evaluate(params):
    x_test = pd.read_csv(params["X_test"])
    y_test = pd.read_csv(params["Y_test"])
    
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

    mlflow.set_experiment("Fit Check Experiment")
    model=pickle.load(open(params["model_path"],'rb'))
        
    run_id = model['metadata']['run_id']
    model = model["model"]
    with mlflow.start_run(run_id=run_id): 
        
        y_pred = model.predict(x_test)

        cm = confusion_matrix(y_test, y_pred)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        mlflow.log_metric("accuracy",accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_text(str(cm),"confusion_matrix.txt")

def main():
    params = yaml.safe_load(open("params.yaml"))["evaluate"]
    evaluate(params)

if __name__=="__main__":
    main()