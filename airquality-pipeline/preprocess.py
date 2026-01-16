import os
import argparse
import logging
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --------------------------------------------------
# Logging configuration
# --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# Arguments
# --------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, default="/opt/ml/processing/input")
    parser.add_argument("--output-data", type=str, default="/opt/ml/processing/output")
    return parser.parse_args()

# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    args = parse_args()
    input_file = os.path.join(args.input_data, "airquality.csv")

    train_dir = os.path.join(args.output_data, "train")
    val_dir = os.path.join(args.output_data, "validation")
    test_dir = os.path.join(args.output_data, "test")
    model_dir = os.path.join(args.output_data, "model")

    for d in [train_dir, val_dir, test_dir, model_dir]:
        os.makedirs(d, exist_ok=True)

    logger.info("Loading dataset")
    df = pd.read_csv(input_file)
    logger.info(f"Initial dataset shape: {df.shape}")

    TARGET_COLUMN = "AQI_Bucket"

    if "Date" in df.columns:
        df.drop(columns=["Date"], inplace=True)
        logger.info("Dropped Date column")

    logger.info("Handling missing values")
    df = df.dropna(subset=[TARGET_COLUMN])

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    logger.info("Encoding target labels")
    label_encoder = LabelEncoder()
    df[TARGET_COLUMN] = label_encoder.fit_transform(df[TARGET_COLUMN])

    if "City" in df.columns:
        df = pd.get_dummies(df, columns=["City"], drop_first=True)
        logger.info("One-hot encoded City column")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    logger.info("Splitting dataset")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.5, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    logger.info("Scaling features")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    logger.info("Saving processed datasets")
    pd.DataFrame(X_train_scaled).to_csv(os.path.join(train_dir, "X_train.csv"), index=False)
    pd.DataFrame(X_val_scaled).to_csv(os.path.join(val_dir, "X_val.csv"), index=False)
    pd.DataFrame(X_test_scaled).to_csv(os.path.join(test_dir, "X_test.csv"), index=False)

    y_train.to_csv(os.path.join(train_dir, "y_train.csv"), index=False)
    y_val.to_csv(os.path.join(val_dir, "y_val.csv"), index=False)
    y_test.to_csv(os.path.join(test_dir, "y_test.csv"), index=False)

    logger.info("Saving preprocessing artifacts")
    joblib.dump(scaler, os.path.join(model_dir, "scaler.joblib"))
    joblib.dump(label_encoder, os.path.join(model_dir, "label_encoder.joblib"))

    logger.info("Preprocessing completed successfully")

'''
if __name__ == "__main__":
    main()
'''