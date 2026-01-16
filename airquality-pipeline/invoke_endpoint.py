import json
import boto3
import logging

# --------------------------------------------------
# Logging
# --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# Configuration
# --------------------------------------------------
REGION = boto3.Session().region_name
ENDPOINT_NAME = "AirQualityEndpoint"   # MUST match deployed endpoint name

runtime = boto3.client("sagemaker-runtime", region_name=REGION)

# --------------------------------------------------
# Payload (example)
# --------------------------------------------------
payload =[ 
{
    "PM2.5": 85,
    "PM10": 120,
    "NO2": 40,
    "SO2": 12,
    "CO": 1.2,
    "O3": 30,
    "City": "Delhi"
},

   {
        "PM2.5": 45,
        "PM10": 70,
        "NO2": 22,
        "SO2": 8,
        "CO": 0.7,
        "O3": 18,
        "City": "Mumbai"
    }
]
# --------------------------------------------------
# Invoke endpoint
# --------------------------------------------------
def invoke_endpoint(payload):
    logger.info(f"Invoking endpoint: {ENDPOINT_NAME}")

    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload)
    )

    result = json.loads(response["Body"].read().decode("utf-8"))
    logger.info(f"Prediction result: {result}")

    return result

# --------------------------------------------------
# Main
# --------------------------------------------------
'''
if __name__ == "__main__":
    invoke_endpoint(payload)
'''