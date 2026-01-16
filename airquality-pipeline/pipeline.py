import sagemaker
import boto3
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
    CreateModelStep,
    CreateEndpointConfigStep,
    CreateEndpointStep,
)
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.parameters import ParameterFloat, ParameterString
from sagemaker.workflow.functions import JsonGet

from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.properties import PropertyFile

# --------------------------------------------------
# Session / Role
# --------------------------------------------------
bucket='main-sagemaker-ml-airquality'
prefix = 'mlops'
session = sagemaker.Session()
role = sagemaker.get_execution_role()
region = session.boto_region_name
#bucket = session.default_bucket()

# --------------------------------------------------
# Pipeline Parameters
# --------------------------------------------------
accuracy_threshold = ParameterFloat(
    name="AccuracyThreshold",
    default_value=0.70
)

input_data = ParameterString(
    name="InputData",
    default_value=f"s3://{bucket}/mlops/airquality.csv"
)

endpoint_name = ParameterString(
    name="EndpointName",
    default_value="airquality-endpoint"
)

# --------------------------------------------------
# Preprocessing Step
# --------------------------------------------------
preprocess_processor = ScriptProcessor(
    image_uri=sagemaker.image_uris.retrieve(
        framework="sklearn",
        region=region,
        version="1.2-1",
    ),
    command=["python3"],
    role=role,
    instance_type="ml.m5.large",
    instance_count=1,
)

preprocess_step = ProcessingStep(
    name="PreprocessData",
    processor=preprocess_processor,
    inputs=[
        ProcessingInput(
            source=input_data,
            destination="/opt/ml/processing/input",
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name="train",
            source="/opt/ml/processing/output/train",
        ),
        ProcessingOutput(
            output_name="validation",
            source="/opt/ml/processing/output/validation",
        ),
        ProcessingOutput(
            output_name="test",
            source="/opt/ml/processing/output/test",
        ),
        ProcessingOutput(
            output_name="preprocess-model",
            source="/opt/ml/processing/output/model",
        ),
    ],
    code="preprocessing.py",
)

# --------------------------------------------------
# Training Step
# --------------------------------------------------
estimator = SKLearn(
    entry_point="train.py",
    role=role,
    instance_type="ml.m5.large",
    instance_count=1,
    framework_version="1.2-1",
    base_job_name="airquality-train",
)

train_step = TrainingStep(
    name="TrainModel",
    estimator=estimator,
    inputs={
        "train": TrainingInput(
            s3_data=preprocess_step.properties
            .ProcessingOutputConfig.Outputs["train"]
            .S3Output.S3Uri
        ),
        "validation": TrainingInput(
            s3_data=preprocess_step.properties
            .ProcessingOutputConfig.Outputs["validation"]
            .S3Output.S3Uri
        ),
        "preprocessing": TrainingInput(
            s3_data=preprocess_step.properties
            .ProcessingOutputConfig.Outputs["preprocess-model"]
            .S3Output.S3Uri
        ),
    },
)

# --------------------------------------------------
# Evaluation Step
# --------------------------------------------------
evaluation_processor = ScriptProcessor(
    image_uri=sagemaker.image_uris.retrieve(
        framework="sklearn",
        region=region,
        version="1.2-1",
    ),
    command=["python3"],
    role=role,
    instance_type="ml.m5.large",
    instance_count=1,
)

evaluation_report = PropertyFile(
    name="EvaluationReport",
    output_name="evaluation",
    path="evaluation.json",
)

evaluation_step = ProcessingStep(
    name="EvaluateModel",
    processor=evaluation_processor,
    inputs=[
        ProcessingInput(
            source=train_step.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/model",
        ),
        ProcessingInput(
            source=preprocess_step.properties
            .ProcessingOutputConfig.Outputs["test"]
            .S3Output.S3Uri,
            destination="/opt/ml/processing/test",
        ),
    ],
    outputs=[
        ProcessingOutput(
            output_name="evaluation",
            source="/opt/ml/processing/evaluation",
        )
    ],
    code="evaluate.py",
    property_files=[evaluation_report],
)

# --------------------------------------------------
# Model Registration
# --------------------------------------------------
sklearn_model = SKLearnModel(
    model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
    role=role,
    entry_point="endpoint.py",
    framework_version="1.2-1",
    sagemaker_session=session,
)

register_step = ModelStep(
    name="RegisterModel",
    step_args=sklearn_model.register(
        model_package_group_name="AirQualityModelGroup",
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.large"],
        approval_status="Approved",
    ),
)

# --------------------------------------------------
# Create Model Step
# --------------------------------------------------
create_model_step = CreateModelStep(
    name="CreateModel",
    model=sklearn_model,
    inputs=sagemaker.inputs.CreateModelInput(
        instance_type="ml.m5.large"
    ),
)

# --------------------------------------------------
# Create Endpoint Config Step
# --------------------------------------------------
endpoint_config_step = CreateEndpointConfigStep(
    name="CreateEndpointConfig",
    endpoint_config_name=f"{endpoint_name}-config",
    model_name=create_model_step.properties.ModelName,
    initial_instance_count=1,
    instance_type="ml.m5.large",
)

# --------------------------------------------------
# Create / Update Endpoint Step
# --------------------------------------------------
endpoint_step = CreateEndpointStep(
    name="DeployEndpoint",
    endpoint_name=endpoint_name,
    endpoint_config_name=endpoint_config_step.properties.EndpointConfigName,
)

# --------------------------------------------------
# Condition Step (Accuracy Gate)
# --------------------------------------------------
condition_step = ConditionStep(
    name="AccuracyCheck",
    conditions=[
        ConditionGreaterThanOrEqualTo(
            left=JsonGet(
                step_name=evaluation_step.name,
                property_file=evaluation_report,
                json_path="accuracy"
            ),
            right=accuracy_threshold
        )
    ],
    if_steps=[
        register_step,
        create_model_step,
        endpoint_config_step,
        endpoint_step
    ],
    else_steps=[]
)


# --------------------------------------------------
# Pipeline Definition
# --------------------------------------------------

pipeline = Pipeline(
    name="AirQualityPipeline",
    parameters=[input_data, accuracy_threshold],
    steps=[preprocess_step,train_step, evaluation_step, condition_step],
    sagemaker_session=session   # Ensure the session is passed here
)

# Create and start the pipeline using SageMaker client--------------

pipeline.create(role_arn=role)  # Replace with your actual role ARN

pipeline.start()

