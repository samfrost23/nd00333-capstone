import json
import numpy as np
import pandas as pd
import os
from sklearn.externals import joblib


def init():
    global model
  #  model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    model = joblib.load("./outputs/model.pkl")

def run(data):
    try:
        request_body = json.loads(data)
        data = pd.DataFrame(request_body["data"])
        result = model.predict(data)
        # You can return any data type, as long as it is JSON serializable.
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error
