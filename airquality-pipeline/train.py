import os
import argparse
import logging
import pandas as pd
import joblib
import shutil

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
# Arguments
# --------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=str, default="/opt/ml/input/data/train")
    parser.add_argument("--val-dir", type=str, default="/opt/ml/input/data/validation")
    parser.add_argument("--prep-dir", type=str, default="/opt/ml/input/data/preprocessing")
    parser.add_argument("--model-dir", type=str, default="/opt/ml/model")
    return parser.parse_args()

# --------------------------------------------------
# Main training logic
# --------------------------------------------------
def main():
    args = parse_args()

    logger.info("Starting training step...")

    # --------------------------------------------------
    # Load training & validation data
    # --------------------------------------------------
    logger.info("Loading training data...")
    X_train = pd.read_csv(os.path.join(args.train_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(args.train_dir, "y_train.csv")).values.ravel()

    logger.info("Loading validation data...")
    X_val = pd.read_csv(os.path.join(args.val_dir, "X_val.csv"))
    y_val = pd.read_csv(os.path.join(args.val_dir, "y_val.csv")).values.ravel()

    logger.info(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")

    # --------------------------------------------------
    # Train model
    # --------------------------------------------------
    logger.info("Training RandomForest model...")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # --------------------------------------------------
    # Validation accuracy
    # --------------------------------------------------
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)

    logger.info(f"Validation Accuracy: {acc:.6f}")

    # --------------------------------------------------
    # Save model
    # --------------------------------------------------
    os.makedirs(args.model_dir, exist_ok=True)

    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path)
    logger.info(f"Model saved at: {model_path}")

    # --------------------------------------------------
    # Copy preprocessing artifacts to model directory
    # --------------------------------------------------
    logger.info("Copying preprocessing artifacts...")

    artifacts = [
        "scaler.joblib",
        "label_encoder.joblib",
        "features.joblib",
    ]

    for artifact in artifacts:
        src = os.path.join(args.prep_dir, artifact)
        dst = os.path.join(args.model_dir, artifact)

        if os.path.exists(src):
            shutil.copy(src, dst)
            logger.info(f"Copied {artifact} → model directory")
        else:
            logger.warning(f"Artifact not found: {src}")

    logger.info("Training step completed successfully ✅")

# --------------------------------------------------
# Entry point
# --------------------------------------------------
if __name__ == "__main__":
    main()
