import os
import json
import logging
import joblib
import pandas as pd

from sklearn.metrics import accuracy_score

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
MODEL_DIR = "/opt/ml/processing/model"
TEST_DIR = "/opt/ml/processing/test"
OUTPUT_DIR = "/opt/ml/processing/evaluation"

# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    logger.info("Starting model evaluation")

    # Load model
    model_path = os.path.join(MODEL_DIR, "model.joblib")
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)

    # Load test data
    X_test = pd.read_csv(os.path.join(TEST_DIR, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(TEST_DIR, "y_test.csv")).values.ravel()

    logger.info(f"Test data shape: X={X_test.shape}, y={y_test.shape}")

    # Predict
    predictions = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, predictions)
    logger.info(f"Test Accuracy: {accuracy:.4f}")

    # Save evaluation report
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    evaluation_path = os.path.join(OUTPUT_DIR, "evaluation.json")

    with open(evaluation_path, "w") as f:
        json.dump({"accuracy": accuracy}, f)

    logger.info(f"Saved evaluation report to {evaluation_path}")


'''if __name__ == "__main__":
    main()'''
