import os
import argparse
import logging
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --------------------------------------------------
# Logging configuration
# --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# Argumentss
# --------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=str, default="/opt/ml/input/data/train")
    parser.add_argument("--val-dir", type=str, default="/opt/ml/input/data/validation")
    parser.add_argument("--model-dir", type=str, default="/opt/ml/model")
    return parser.parse_args()

# --------------------------------------------------
# Main training logic
# --------------------------------------------------
def main():
    args = parse_args()

    # Load training data
    logger.info("Loading training data")
    X_train = pd.read_csv(os.path.join(args.train_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(args.train_dir, "y_train.csv")).values.ravel()

    # Load validation data
    logger.info("Loading validation data")
    X_val = pd.read_csv(os.path.join(args.val_dir, "X_val.csv"))
    y_val = pd.read_csv(os.path.join(args.val_dir, "y_val.csv")).values.ravel()

    # Train model
    logger.info("Training RandomForest model")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Validation accuracy
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)

    logger.info(f"Validation Accuracy: {acc:.4f}")

    # Save model
    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path)

    logger.info(f"Model saved to {model_path}")

# --------------------------------------------------
# Entry point
# --------------------------------------------

if __name__ == "__main__":
    main()

import shutil

# Copy preprocessing artifacts into model directory
shutil.copy(
    "/opt/ml/input/data/preprocessing/model/scaler.joblib",
    os.path.join(args.model_dir, "scaler.joblib")
)

shutil.copy(
    "/opt/ml/input/data/preprocessing/model/label_encoder.joblib",
    os.path.join(args.model_dir, "label_encoder.joblib")
)
