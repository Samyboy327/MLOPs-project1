import json 
import joblib
import pandas as pd
import os

def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
    encoder = joblib.load(os.path.join(model_dir, "label_encoder.joblib"))
    return model, scaler, encoder

def input_fn(request_body, content_type):
    if content_type == "application/json":
        return pd.DataFrame(json.loads(request_body))
    raise ValueError("Unsupported content type")

def predict_fn(input_data, model_artifacts):
    model, scaler, encoder = model_artifacts
    X_scaled = scaler.transform(input_data)
    preds = model.predict(X_scaled)
    return encoder.inverse_transform(preds)

def output_fn(prediction, accept):
    return json.dumps({"predictions": prediction.tolist()}), accept
