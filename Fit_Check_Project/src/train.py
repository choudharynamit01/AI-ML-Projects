from sklearn.model_selection import train_test_split, GridSearchCV
import yaml
import pandas as pd
import pickle
import os
import mlflow
from sklearn.linear_model import LogisticRegression
from mlflow.models import infer_signature
from urllib.parse import urlparse

os.environ['MLFLOW_TRACKING_URI']="http://127.0.0.1:5000"

def train(params):
    data = pd.read_csv(params["input"])
    
    x = data.drop(columns='is_fit')
    y = data['is_fit']
    
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

    mlflow.set_experiment("Fit Check Experiment")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        X_train, X_test, y_train, y_test = train_test_split(
                x, y,
                test_size=0.2,        # 20% test, 80% train
                random_state=42,      # for reproducibility
                stratify=y            # optional: keeps target class distribution balanced
            )
        signature = infer_signature(X_train, y_train)
        model = LogisticRegression()
        model.fit(X_train, y_train)
        
        tracking_url_type_store=urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store!='file':
            mlflow.sklearn.log_model(model,"model",registered_model_name="Fit Check Beast", signature=signature)
        else:
            mlflow.sklearn.log_model(model, "model",signature=signature)

        os.makedirs(os.path.dirname(params["output"]),exist_ok=True)
        filename=params["output"]
        model_with_metadata = {
        'model': model,
        'metadata': {'run_id': run_id}
       }
        pickle.dump(model_with_metadata,open(filename,'wb'))

        os.makedirs(os.path.dirname(params["X_test"]), exist_ok=True)
        X_test.to_csv(params["X_test"], index=False)

        os.makedirs(os.path.dirname(params["y_test"]), exist_ok=True)
        y_test.to_csv(params["y_test"], index=False)

        print(f"Model saved to {filename}")


def main():
    params = yaml.safe_load(open("params.yaml"))["train"]
    train(params)

if __name__=="__main__":
    main()