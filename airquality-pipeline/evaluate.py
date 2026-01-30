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
# Paths (SageMaker standard)
# --------------------------------------------------
MODEL_DIR = "/opt/ml/processing/model"
TEST_DIR = "/opt/ml/processing/test"
OUTPUT_DIR = "/opt/ml/processing/evaluation"

# --------------------------------------------------
# Main Evaluation Logic
# --------------------------------------------------
def main():
    logger.info("Starting model evaluation...")

    # Load model
    model_path = os.path.join(MODEL_DIR, "model.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    logger.info(f"Loading model from: {model_path}")
    model = joblib.load(model_path)

    # Load test data
    X_test_path = os.path.join(TEST_DIR, "X_test.csv")
    y_test_path = os.path.join(TEST_DIR, "y_test.csv")

    logger.info(f"Loading test data from: {TEST_DIR}")

    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).values.ravel()

    logger.info(f"Test data shape -> X: {X_test.shape}, y: {y_test.shape}")

    # Predictions
    predictions = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, predictions)
    logger.info(f"Test Accuracy: {accuracy:.6f}")

    # Save evaluation report (IMPORTANT for ConditionStep)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    evaluation_path = os.path.join(OUTPUT_DIR, "evaluation.json")

    report = {
        "accuracy": float(accuracy)
    }

    with open(evaluation_path, "w") as f:
        json.dump(report, f)

    logger.info(f"Evaluation report saved at: {evaluation_path}")
    logger.info(f"Evaluation JSON: {report}")

# --------------------------------------------------
# Entry Point (CRITICAL)
# --------------------------------------------------
if __name__ == "__main__":
    main()
