import os
import json
import logging
import joblib
import pandas as pd

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
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.joblib")

# --------------------------------------------------
# Load model artifacts
# --------------------------------------------------
def model_fn(model_dir):
    logger.info("Loading model artifacts")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)

    logger.info("Model, scaler, and encoder loaded successfully")

    return {
        "model": model,
        "scaler": scaler,
        "encoder": encoder
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
    logger.info("Running inference")

    model = artifacts["model"]
    scaler = artifacts["scaler"]
    encoder = artifacts["encoder"]

    # Columns used during training
    numeric_cols = scaler.feature_names_in_
    categorical_cols = encoder.feature_names_in_

    # Split features
    X_num = input_data[numeric_cols]
    X_cat = input_data[categorical_cols]

    # Transform
    X_num_scaled = scaler.transform(X_num)
    X_cat_encoded = encoder.transform(X_cat)

    # Combine
    X_final = pd.concat(
        [
            pd.DataFrame(X_num_scaled, columns=numeric_cols),
            pd.DataFrame(
                X_cat_encoded,
                columns=encoder.get_feature_names_out(categorical_cols)
            )
        ],
        axis=1
    )

    predictions = model.predict(X_final)

    return predictions

# --------------------------------------------------
# Output handler
# --------------------------------------------------
def output_fn(predictions, accept):
    logger.info("Formatting response")

    if accept == "application/json":
        return json.dumps({
            "predictions": predictions.tolist()
        }), accept

    raise ValueError(f"Unsupported accept type: {accept}")
