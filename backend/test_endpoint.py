from botocore.config import Config
import boto3
import json

config = Config(read_timeout=300, connect_timeout=60)

runtime = boto3.client(
    "sagemaker-runtime",
    region_name="ap-south-1",
    config=config
)

response = runtime.invoke_endpoint(
    EndpointName="sentiment-analysis-endpoint-v7",
    ContentType="application/json",
    Body=json.dumps({
        "video_path": "s3://multimodal-sentiment-analysis-iiit-bbsr-mumbai/inference/sample.mp4"
    })
)

print(response["Body"].read().decode())