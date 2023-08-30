from pipelines._utils import read_config, find_latest_approved_model_url
import os
import boto3
import json
import sagemaker
import sagemaker.session
import configparser

from sagemaker.inputs import TrainingInput, CreateModelInput, TransformInput
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.transformer import Transformer
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.condition_step import (
    ConditionStep,
)
from sagemaker.workflow.functions import (
    JsonGet,
)
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
    ParameterBoolean,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
    TransformStep
)
from sagemaker.workflow.model_step import ModelStep
from sagemaker.model import Model
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.amazon.amazon_estimator import get_image_uri

from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.transformer import Transformer
from sagemaker.inputs import BatchDataCaptureConfig


BASE_DIR = os.path.dirname(os.path.realpath(__file__))

config_bucket = 'pipeline-aiml-stg'
config_key = 'OutOfStock/Configurations/configurations.ini'
config_uri = f"s3://{config_bucket}/{config_key}"

# Initialize the Parser object
config = configparser.ConfigParser()
config_string = read_config(bucket=config_bucket, key=config_key)
config.read_string(config_string.decode("utf-8"))

s3_bucket = config.get('OutOfStock_Properties', 's3_bucket')
s3_prefix = config.get('OutOfStock_Properties', 's3_prefix')
data_capture_prefix = config.get('OutOfStock_Properties', 'data_capture_prefix')

data_capture_destination = f"s3://{s3_bucket}/{s3_prefix}/{data_capture_prefix}"

model_url = str(find_latest_approved_model_url('OutOfStockPackageGroup'))
print(f"model_url: {model_url}")


def get_sagemaker_client(region):
    """Gets the sagemaker client.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    return sagemaker_client


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )


def get_pipeline_session(region, default_bucket):
    """Gets the pipeline session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        PipelineSession instance
    """

    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")

    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )


def get_pipeline_custom_tags(new_tags, region, sagemaker_project_arn=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.list_tags(
            ResourceArn=sagemaker_project_arn.lower())
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags


def get_pipeline(
        region,
        sagemaker_project_arn=None,
        role=None,
        default_bucket=None,
        model_package_group_name="OutOfStockDriftCheckGroup",
        pipeline_name="OutOfStockDriftCheckpipeline",
        base_job_prefix="OutOfStockDriftCheck",
        processing_instance_type="ml.m5.large",
        # processing_instance_type="ml.t3.medium",
        training_instance_type="ml.m4.xlarge"
):
    """Gets a SageMaker ML Pipeline instance working with on OutOfStock data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
    sagemaker_session = get_session(region, default_bucket)

    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    pipeline_session = get_pipeline_session(region, default_bucket)

    # parameters for pipeline execution
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    register_new_model_baseline = ParameterString(name="RegisterNewBaseline", default_value="False")

    # Create pipeline step 
    image_uri = sagemaker.image_uris.retrieve(
        framework="forecasting-deepar",
        region=region,
        version="1.0-1",
        py_version="py3",
    )

    model = Model(
        image_uri=image_uri,
        model_data=model_url,
        role=role,
        sagemaker_session=pipeline_session,
    )

    step_create_oos_model = ModelStep(
        name="CreateOutOfStockModelStep",
        step_args=model.create(),
    )

    #  Process Ground Truth 
    sklearn_processor = SKLearnProcessor(
        framework_version="0.23-1",
        instance_type="ml.m5.xlarge",
        instance_count=1,
        base_job_name=f"{base_job_prefix}/sklearn-outofstock-creategt",
        sagemaker_session=pipeline_session,
        role=role,
    )
    step_args = sklearn_processor.run(
        inputs=[
            ProcessingInput(source=config_uri, destination="/opt/ml/processing/config/"),
        ],
        outputs=[
            ProcessingOutput(output_name="TranformInput", source="/opt/ml/processing/TransformInput/"),
        ],
        code=os.path.join(BASE_DIR, "process_oos_gt.py"),
        # arguments=["--input-data", input_data],
    )
    step_create_oos_gt = ProcessingStep(
        name="CreateOutofStockGroundTruth",
        step_args=step_args,
    )

    #  Batch Transform step for generating batch predictions
    # assume we only check '0.5' quatiles predictions.
    environment_param = {
        'num_samples': 20,
        'output_types': ['quantiles'],
        'quantiles': ['0.5']
    }

    transformer = Transformer(
        model_name=step_create_oos_model.properties.ModelName,
        instance_type="ml.m5.xlarge",
        instance_count=1,
        output_path=f's3://{s3_bucket}/{s3_prefix}/BATCH_TRANSFORM/OUTPUT/',
        sagemaker_session=pipeline_session,
        accept='application/json',
        strategy='MultiRecord',
        assemble_with='Line',
        env={
            'DEEPAR_INFERENCE_CONFIG': json.dumps(environment_param)
        },
    )

    transform_inputs = TransformInput(
        data=step_create_oos_gt.properties.ProcessingOutputConfig.Outputs["TranformInput"].S3Output.S3Uri,

    )

    step_args = transformer.transform(
        data=transform_inputs.data,
        #     batch_data_capture_config=BatchDataCaptureConfig(
        #     destination_s3_uri=data_capture_destination,
        #     # kms_key_id="kms_key",
        #     generate_inference_id=True
        # ),
        # model_name = step_create_model.properties.ModelName,
        # input_filter="$[1:]",
        # join_source="Input",
        # output_filter="$[0,-1]",
        # content_type="text/json",
        split_type="Line",
    )

    step_outofstock_transform = TransformStep(
        name="OutOfStockBatchTransform",
        step_args=step_args,
    )

    #  Custom Model Quality Check
    sklearn_processor = SKLearnProcessor(
        framework_version="0.23-1",
        instance_type="ml.m5.xlarge",
        instance_count=1,
        base_job_name=f"{base_job_prefix}/sklearn-outofstock-model-quality-check",
        sagemaker_session=pipeline_session,
        role=role,
        env={"register_new_model_baseline": register_new_model_baseline}
    )

    step_args = sklearn_processor.run(
        inputs=[
            ProcessingInput(
                source=step_create_oos_gt.properties.ProcessingOutputConfig.Outputs["TranformInput"].S3Output.S3Uri,
                destination="/opt/ml/processing/TransformInput/",
            ),
            ProcessingInput(
                source=step_outofstock_transform.properties.TransformOutput.S3OutputPath,
                destination="/opt/ml/processing/TransformOutput/",
            ),
            ProcessingInput(source=config_uri, destination="/opt/ml/processing/config/"),
        ],
        outputs=[
            ProcessingOutput(output_name="Statistics", source="/opt/ml/processing/statistics.json"),
            ProcessingOutput(output_name="Constraints", source="/opt/ml/processing/constraints.json"),
            ProcessingOutput(output_name="Constraints_Violations",
                             source="/opt/ml/processing/constraints_violations.json"),
        ],
        code=os.path.join(BASE_DIR, "CustomModelQualityCheck.py"),
        arguments=["RegisterNewBaseline", register_new_model_baseline],
    )

    step_model_quality_check = ProcessingStep(
        name="CustomModelQualityCheck",
        step_args=step_args,
    )

    # pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            register_new_model_baseline,

        ],
        steps=[step_create_oos_model, step_create_oos_gt, step_outofstock_transform, step_model_quality_check],
        sagemaker_session=pipeline_session,
    )
    return pipeline
