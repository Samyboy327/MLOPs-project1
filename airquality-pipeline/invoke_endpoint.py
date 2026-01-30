import boto3
import json

endpoint_name = "airquality-endpoint"

runtime = boto3.client("sagemaker-runtime")

payload = [
    {
        "PM2.5": 80,
        "PM10": 120,
        "NO2": 40,
        "SO2": 10,
        "CO": 1.2,
        "O3": 30
    }
]

response = runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType="application/json",
    Body=json.dumps(payload),
)

print(response["Body"].read().decode())
