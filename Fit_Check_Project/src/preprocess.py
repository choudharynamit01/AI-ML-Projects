import os
import yaml
import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder


def preprocess(params):
    data = pd.read_csv(params["input"])

    x = data.drop(columns='is_fit')
    y = data['is_fit']

    x['age'] = pd.cut(x['age'], bins=params["age_bins"], labels=params["age_labels"], right=False)   

    x["smokes"] = x["smokes"].apply(does_smoke)

    
    ordinal_encoder = OrdinalEncoder(categories=[params["age_labels"]])
    x['age'] = ordinal_encoder.fit_transform(x[['age']])

    onehot_encoder = OneHotEncoder(sparse_output=False, drop='first')  # drop first to avoid dummy variable trap
    x['gender'] = onehot_encoder.fit_transform(x[['gender']])

    imputer = IterativeImputer(estimator=BayesianRidge(),  # You can change this
                           max_iter=params["max_iter"],
                           random_state=params["random_state"])
    
    imputed_data = imputer.fit_transform(x)
    imputed_data = pd.DataFrame(imputed_data, columns = x.columns)
    imputed_data[y.name] = y
    os.makedirs(os.path.dirname(params["output"]), exist_ok=True)
    imputed_data.to_csv(params["output"], index=False)

def does_smoke(smoke):
        if smoke == "yes":
            return 1
        elif smoke == "no":
            return 0

        return smoke 
    

def main():
    params = yaml.safe_load(open("params.yaml"))["data"]
    preprocess(params)


if __name__ == '__main__':
    main()
