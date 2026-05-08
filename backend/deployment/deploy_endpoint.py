import os
import boto3
from sagemaker.pytorch import PyTorchModel
import sagemaker


def deploy_endpoint():
    print(" Starting deployment...")

    boto_session = boto3.Session(region_name="ap-south-1")
    sagemaker_session = sagemaker.Session(boto_session=boto_session)

    role = "arn:aws:iam::017820698547:role/sentiment-analysis-deploy-endpoint-role-mumbai"

    model_uri = "s3://multimodal-sentiment-analysis-iiit-bbsr-mumbai/inference/model.tar.gz"

    model = PyTorchModel(
        model_data=model_uri,
        role=role,
        framework_version="2.5.1",
        py_version="py311",
        entry_point="inference.py",
        source_dir="../sagemaker_package/code",
        name="sentiment-analysis-model-v7",
        sagemaker_session=sagemaker_session
    )

    print(" Model object created")

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type="ml.g4dn.xlarge",
        endpoint_name="sentiment-analysis-endpoint-v7",
    )

    print(" Endpoint deployed successfully!")


if __name__ == "__main__":
    deploy_endpoint()
