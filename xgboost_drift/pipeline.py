"""Drift check pipeline for Model monitoring"""
from pipelines._utils import read_config, find_latest_approved_model_url
import configparser
import boto3
import os
import logging

import sagemaker
from sagemaker.network import NetworkConfig
from sagemaker.workflow.pipeline import Pipeline
from sagemaker import get_execution_role, session
from sagemaker.model import Model
from sagemaker.model_monitor.dataset_format import DatasetFormat
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
    TuningStep,
    CacheConfig,
    TransformStep
)
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.transformer import Transformer
from sagemaker.workflow.model_step import ModelStep
from sagemaker.inputs import TrainingInput, CreateModelInput, TransformInput, BatchDataCaptureConfig
from sagemaker.workflow.check_job_config import CheckJobConfig
from sagemaker.workflow.quality_check_step import ModelQualityCheckConfig
from sagemaker.workflow.monitor_batch_transform_step import MonitorBatchTransformStep
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
    ParameterBoolean,
)
from sagemaker.workflow.quality_check_step import (
    DataQualityCheckConfig,
    ModelQualityCheckConfig,
    QualityCheckStep,
)
from sagemaker.workflow.functions import Join

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

# Configuration File
config_bucket = 'pipeline-aiml-stg'
config_key = 'BCM_Pipeline_Data/Configurations/configurations.ini'
config_uri = f"s3://{config_bucket}/{config_key}"

# Initialize the Parser object
config = configparser.ConfigParser()
config_string = read_config(bucket=config_bucket ,key=config_key)
config.read_string(config_string.decode("utf-8"))


# bucket = session.Session(boto3.Session()).default_bucket()
bucket = 'pipeline-aiml-stg'

data_capture_prefix = config.get('BCM_Properties', 'data_capture_prefix')
s3_capture_upload_path = "s3://{}/{}".format(bucket, data_capture_prefix)

reports_prefix = config.get('BCM_Properties', 's3_report_path_prefix')
s3_report_path = "s3://{}/{}".format(bucket, reports_prefix)

transform_prefix = config.get('BCM_Properties', 'transform_prefix')
transform_output_path = "s3://{}/{}".format(bucket, transform_prefix )

data_monitor_reports_prefix = config.get('BCM_Properties', 'data_monitor_reports_prefix')
s3_data_report_path = "s3://{}/{}".format(bucket, data_monitor_reports_prefix )
baseline_data_results_prefix = config.get('BCM_Properties', 'baseline_data_results_prefix')

print("S3 storage Paths")
print("Transform Output path: {}".format(transform_output_path))
print("Capture path: {}".format(s3_capture_upload_path))
print("Report path: {}".format(s3_report_path))


# Model Baselining Paths
baseline_results_prefix = config.get('BCM_Properties', 'baseline_results_prefix')
baseline_results_uri = "s3://{}/{}".format(bucket, baseline_results_prefix)
print("Baseline results uri:",baseline_results_uri)
print(f"baseline_data_results_prefix : {baseline_data_results_prefix}")


# Data Baselining 
supplied_baseline_statistics_data_quality = "s3://{}/{}/statistics.json".format(bucket, baseline_data_results_prefix)
supplied_baseline_constraints_data_quality = "s3://{}/{}/constraints.json".format(bucket, baseline_data_results_prefix)  
print(f"supplied_baseline_statistics_data_quality: {supplied_baseline_statistics_data_quality}")
print(f"supplied_baseline_constraints_data_quality: {supplied_baseline_constraints_data_quality}")

model_url = str(find_latest_approved_model_url('BcmPackageGroup'))
print(f"model_url : {model_url}")


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
        model_package_group_name="BCMDriftCheckPackageGroup",
        pipeline_name="BCMDriftCheckPipeline",
        base_job_prefix="BCMDriftCheck",
        processing_instance_type="ml.m5.large",
        # processing_instance_type="ml.t3.medium",
        training_instance_type="ml.m4.xlarge",
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

    image_uri = sagemaker.image_uris.retrieve(
        framework="xgboost",
        region=region,
        version="1.0-1",
        py_version="py3",
        # instance_type=training_instance_type,
    )
    print(image_uri)

    model = Model(
        image_uri=image_uri,
        model_data=model_url,
        role=role,
        sagemaker_session=pipeline_session,
    )

    step_create_model = ModelStep(
        name="CreateXGBoostModelStep",
        step_args=model.create(),
    )

    # Create Ground Truth Step
    base_job_prefix = 'BCM_DriftCheck'

    # network_config = NetworkConfig (
    #     encrypt_inter_container_traffic=True,
    #     security_group_ids=["sg-084b4d82306252edc"],
    #     subnets=["subnet-014410aa0c935e9bc","subnet-00df143e1661fce55","subnet-03b7c99200115eb72"]
    # )

    
    sklearn_processor = SKLearnProcessor(
        framework_version="0.23-1",
        instance_type="ml.m5.xlarge",
        instance_count=1,
        base_job_name=f"{base_job_prefix}/sklearn-bcm-creategt",
        sagemaker_session=pipeline_session,
        role=role,
        # network_config=network_config,
        # image_uri= image_uri,
    )
    step_args = sklearn_processor.run(
         inputs=[
            ProcessingInput(source=config_uri, destination="/opt/ml/processing/config"),
        ],
        outputs=[
            ProcessingOutput(output_name="Test", source="/opt/ml/processing/Test/"), 
            ProcessingOutput(output_name="TranformInput", source="/opt/ml/processing/TransformInput/"),        
        ],
        code=os.path.join(BASE_DIR, "process_gt.py"),
    )
    step_create_gt = ProcessingStep(
        name="CreateBCMGroundTruth",
        step_args=step_args,
    )

    #  Data Quality Check Step
    check_job_config = CheckJobConfig(
    role=role,
    instance_count=1,
    instance_type="ml.c5.xlarge",
    volume_size_in_gb=120,
    sagemaker_session=pipeline_session,
    )
    
    data_quality_check_config = DataQualityCheckConfig(      baseline_dataset=step_create_gt.properties.ProcessingOutputConfig.Outputs["Test"].S3Output.S3Uri,
        dataset_format=DatasetFormat.csv(header=True, output_columns_position="START"), #SECOND
        output_s3_uri = Join(on='/', values=['s3:/', bucket, data_monitor_reports_prefix ]),   
    )

    data_quality_check_step = QualityCheckStep(
        name="DataQualityCheckStep",
        skip_check=False,
        fail_on_violation=True,
        register_new_baseline=False,
        quality_check_config=data_quality_check_config,
        check_job_config=check_job_config,
        supplied_baseline_statistics = supplied_baseline_statistics_data_quality,
        supplied_baseline_constraints = supplied_baseline_constraints_data_quality,
        model_package_group_name=model_package_group_name
    )
    
    
    
    # Create Monitor and Batch Transform Step

    # Configure a Transformer
    transformer = Transformer(
        model_name=step_create_model.properties.ModelName,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        accept="text/csv",
        assemble_with="Line",
        output_path=transform_output_path,
        sagemaker_session=pipeline_session,
    )

    transform_inputs = TransformInput(
        data=step_create_gt.properties.ProcessingOutputConfig.Outputs["TranformInput"].S3Output.S3Uri,
    )

    transform_arg = transformer.transform(
        transform_inputs.data,
        content_type="text/csv",
        split_type="Line",
         # exclude the ground truth (first column) from the validation set
        # when doing inference.  
        input_filter="$[1:]",
        join_source="Input",
        output_filter="$[0,-1]",           
        batch_data_capture_config=BatchDataCaptureConfig(
            destination_s3_uri=s3_capture_upload_path,
            generate_inference_id=True, )
    )
      

    # Configure Model Quality Check
    job_config = CheckJobConfig(role=role)

    model_quality_config = ModelQualityCheckConfig(
        # The dataset we want to run evaluation against
        # in this example, this is the same as the transform input
        baseline_dataset=transformer.output_path,
        problem_type="Regression",
        dataset_format=DatasetFormat.csv(header=False),
        output_s3_uri=s3_report_path,
        # since we joined the transform input and output, the output will be
        # following the input. There are 1 column for ground truth, and 69 for input features
        # so the index (0-based) for the output (inference prediction) will be 70
        inference_attribute='_c1',
        # remember that the ground truth in the validation set is the ground truth
        ground_truth_attribute="_c0",
    )

    constraints_path = f"{baseline_results_uri}/constraints.json"
    statistics_path = f"{baseline_results_uri}/statistics.json"

    transform_and_monitor_step = MonitorBatchTransformStep(
        name="MonitorModelQuality",
        transform_step_args=transform_arg,
        monitor_configuration=model_quality_config,
        check_job_configuration=job_config,
        # Since when we are generating the baselines using ground truth
        # any test data, there are violations for sure. Let's
        # fail the pipeline execution.
        fail_on_violation=True,
        supplied_baseline_constraints=constraints_path,
        supplied_baseline_statistics=statistics_path
    )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            training_instance_type,
        ],
        steps= [step_create_model, step_create_gt, data_quality_check_step, transform_and_monitor_step],
        sagemaker_session=pipeline_session,
    )
    return pipeline











