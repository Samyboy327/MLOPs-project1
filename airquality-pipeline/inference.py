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
# Paths (SageMaker standard)
# --------------------------------------------------
MODEL_DIR = "/opt/ml/model"

MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.joblib")

# --------------------------------------------------
# Load artifacts
# --------------------------------------------------
def model_fn(model_dir):
    logger.info("Loading model artifacts")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)

    logger.info("Model, scaler, encoder loaded successfully")

    return {
        "model": model,
        "scaler": scaler,
        "encoder": encoder
    }

# --------------------------------------------------
# Parse input
# --------------------------------------------------
def input_fn(request_body, content_type):
    logger.info(f"Received request with content type: {content_type}")

    if content_type == "application/json":
        data = json.loads(request_body)

        # Expect either single record or list
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = pd.DataFrame(data)

        return df

    raise ValueError(f"Unsupported content type: {content_type}")

# --------------------------------------------------
# Prediction
# --------------------------------------------------
def predict_fn(input_data, model_artifacts):
    logger.info("Running inference")

    model = model_artifacts["model"]
    scaler = model_artifacts["scaler"]
    encoder = model_artifacts["encoder"]

    # Separate categorical & numerical columns
    cat_cols = encoder.feature_names_in_
    num_cols = scaler.feature_names_in_

    X_cat = input_data[cat_cols]
    X_num = input_data[num_cols]

    # Transform
    X_cat_encoded = encoder.transform(X_cat)
    X_num_scaled = scaler.transform(X_num)

    # Combine features
    X_final = pd.concat(
        [
            pd.DataFrame(X_num_scaled, columns=num_cols),
            pd.DataFrame(
                X_cat_encoded,
                columns=encoder.get_feature_names_out(cat_cols)
            )
        ],
        axis=1
    )

    predictions = model.predict(X_final)

    return predictions

# --------------------------------------------------
# Output formatting
# --------------------------------------------------
def output_fn(predictions, accept):
    logger.info("Formatting output")

    if accept == "application/json":
        return json.dumps({
            "predictions": predictions.tolist()
        }), accept

    raise ValueError(f"Unsupported accept type: {accept}")
