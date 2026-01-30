import os
import json
import logging
import joblib
import pandas as pd
import numpy as np

# --------------------------------------------------
# Logging
# --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# Paths
# --------------------------------------------------
MODEL_DIR = "/opt/ml/model"

MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.joblib")
FEATURES_PATH = os.path.join(MODEL_DIR, "features.joblib")

# --------------------------------------------------
# Load model artifacts
# --------------------------------------------------
def model_fn(model_dir):
    logger.info("Loading model artifacts...")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    features = joblib.load(FEATURES_PATH)

    logger.info("Artifacts loaded successfully ✅")

    return {
        "model": model,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "features": features,
    }

# --------------------------------------------------
# Input handler
# --------------------------------------------------
def input_fn(request_body, content_type):
    logger.info(f"Content type: {content_type}")

    if content_type == "application/json":
        data = json.loads(request_body)

        if isinstance(data, dict):
            return pd.DataFrame([data])
        elif isinstance(data, list):
            return pd.DataFrame(data)

    raise ValueError(f"Unsupported content type: {content_type}")

# --------------------------------------------------
# Prediction logic
# --------------------------------------------------
def predict_fn(input_data, artifacts):
    logger.info("Running inference...")

    model = artifacts["model"]
    scaler = artifacts["scaler"]
    label_encoder = artifacts["label_encoder"]
    features = artifacts["features"]

    # Ensure all expected features exist
    for col in features:
        if col not in input_data.columns:
            input_data[col] = 0  # default for missing one-hot columns

    # Reorder columns to match training
    input_data = input_data[features]

    # Scale numeric features
    X_scaled = scaler.transform(input_data)

    # Predict
    preds_encoded = model.predict(X_scaled)

    # Decode labels
    preds = label_encoder.inverse_transform(preds_encoded)

    return preds

# --------------------------------------------------
# Output handler
# --------------------------------------------------
def output_fn(predictions, accept):
    logger.info("Formatting response...")

    if accept == "application/json":
        return json.dumps({
            "predictions": predictions.tolist()
        }), accept

    raise ValueError(f"Unsupported accept type: {accept}")
