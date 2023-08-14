"""Example workflow pipeline script for OutOfStock pipeline.

                                               . -ModelStep
                                              .
    Process-> Train -> Evaluate -> Condition .
                                              .
                                               . -(stop)

Implements a get_pipeline(**kwargs) method.
"""
import os
import boto3
import json
import sagemaker
import sagemaker.session
import ast
import configparser
import pathlib

from sagemaker.estimator import Estimator
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
from sagemaker.model_monitor import DatasetFormat, model_monitoring
from sagemaker.workflow.clarify_check_step import (
    DataBiasCheckConfig,
    ClarifyCheckStep,
    ModelBiasCheckConfig,
    ModelPredictedLabelConfig,
    ModelExplainabilityCheckConfig,
    SHAPConfig
)
from sagemaker.workflow.quality_check_step import (
    DataQualityCheckConfig,
    ModelQualityCheckConfig,
    QualityCheckStep,
)
from sagemaker.workflow.functions import Join
from sagemaker.drift_check_baselines import DriftCheckBaselines

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

config_bucket = 'sagemaker-eu-west-1-610914939903'
config_key = 'oos_prediction/OutOfStock/Configurations/configurations.ini'
config_uri = f"s3://{config_bucket}/{config_key}"


def read_config(bucket, key, profile_name=None):
    session = boto3.Session()
    s3 = session.client('s3')
    s3_object = s3.get_object(Bucket=bucket, Key=key)
    body = s3_object['Body']

    #  Upload to local path   
    base_dir = "/opt/ml/config"
    pathlib.Path(f"{base_dir}").mkdir(parents=True, exist_ok=True)
    fn = f"{base_dir}/configurations.ini"
    s3 = boto3.resource("s3")
    s3.Bucket(bucket).download_file(key, fn)
    return body.read()


# Initialize the Parser object
config = configparser.ConfigParser()
config_string = read_config(bucket=config_bucket, key=config_key)
config.read_string(config_string.decode("utf-8"))

s3_bucket = config.get('OutOfStock_Properties', 's3_bucket')
s3_prefix = config.get('OutOfStock_Properties', 's3_prefix')
input_prefix = config.get('OutOfStock_Properties', 'input_prefix')
input_prefix_QA = config.get('OutOfStock_Properties', 'input_prefix_QA') #added for QA env
data_capture_prefix = config.get('OutOfStock_Properties', 'data_capture_prefix')
transform_prefix = config.get('OutOfStock_Properties', 'transform_prefix')
hyperparameters_dict = config.get('OutOfStock_Model_Properties', 'hyperparameters_dict')
custom_image_uri = config.get('OutOfStock_Model_Properties', 'custom_image_uri')

base_dir = f"s3://{s3_bucket}/{s3_prefix}"
input_path = f"s3://{s3_bucket}/{s3_prefix}/{input_prefix}"
# input_path = f"s3://{s3_bucket}/{s3_prefix}/{input_prefix_QA}"  #added for QA env
data_capture_destination = f"s3://{s3_bucket}/{s3_prefix}/{data_capture_prefix}"
transform_ouput_path = f"s3://{s3_bucket}/{s3_prefix}/{transform_prefix}/OUTPUT/"
transform_input_path = f"s3://{s3_bucket}/{s3_prefix}/{transform_prefix}/INPUT/batch_transform_input.json"

hyperparameters = ast.literal_eval(hyperparameters_dict)
print(f"Hyperparamters : {hyperparameters} ")


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
        model_package_group_name="OOSPackageGroup",
        pipeline_name="OOSPipeline",
        base_job_prefix="OOS",
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
    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )
    # parameter for input data
    input_data = ParameterString(
        name="input_data",
        default_value=input_path,
    )

    # processing step for feature engineering
    sklearn_processor = SKLearnProcessor(
        framework_version="0.23-1",
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}/sklearn-OutOfStock-preprocess",
        sagemaker_session=pipeline_session,
        role=role,
    )
    step_args = sklearn_processor.run(
        inputs=[
            ProcessingInput(source=config_uri, destination="/opt/ml/processing/config"),
            ProcessingInput(source=input_data, destination="/opt/ml/processing/input_data"),
        ],
        outputs=[
            ProcessingOutput(output_name="train", source=f"/opt/ml/processing/train.json"),
            ProcessingOutput(output_name="test", source=f"/opt/ml/processing/test.json"),
        ],
        code=os.path.join(BASE_DIR, "preprocess.py"),
        arguments=["--input-data", input_data],
    )
    step_process = ProcessingStep(
        name="PreprocessOutOfStockData",
        step_args=step_args,
    )

    # training step for generating model artifacts
    model_uri = f"s3://{sagemaker_session.default_bucket()}/{base_job_prefix}/OutOfStockTrain"
    image_uri = sagemaker.image_uris.retrieve(
        framework="forecasting-deepar",
        region=sagemaker_session.boto_region_name,
        version="1.0-1",
        py_version="py3",
        instance_type=training_instance_type,
    )

    deepar_train = sagemaker.estimator.Estimator(image_uri,
                                                 role=role,
                                                 train_instance_count=1,
                                                 train_instance_type="ml.m4.xlarge",
                                                 base_job_name=f"{base_job_prefix}/OutOfStock-train",
                                                 output_path=f"{base_dir}/OUTPUT",
                                                 sagemaker_session=pipeline_session,
                                                 )
    deepar_train.set_hyperparameters(**hyperparameters)

    step_args = deepar_train.fit(inputs={
        'train': f'{base_dir}/TRAIN/train.json',
        'test': f'{base_dir}/TEST/test.json'
    })

    step_train = TrainingStep(
        name="TrainOutOfStockModel",
        step_args=step_args
    )
    step_train.add_depends_on([step_process])

    # Create Model step
    model = Model(
        image_uri=image_uri,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        sagemaker_session=pipeline_session,
        role=role,
    )
    step_args = model.create(
        instance_type="ml.m5.large",
        accelerator_type="ml.eia1.medium",
    )
    step_create_model = ModelStep(
        name="OutOfStockCreateModel",
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
        model_name=step_create_model.properties.ModelName,
        instance_type="ml.m5.xlarge",
        instance_count=1,
        output_path=transform_ouput_path,
        sagemaker_session=pipeline_session,
        accept='application/json',
        strategy='MultiRecord',
        assemble_with='Line',
        env={
            'DEEPAR_INFERENCE_CONFIG': json.dumps(environment_param)
        },
    )

    transform_inputs = TransformInput(data=transform_input_path, )
    step_args = transformer.transform(
        data=transform_inputs.data,
        batch_data_capture_config=BatchDataCaptureConfig(
            destination_s3_uri=data_capture_destination,
            generate_inference_id=True
        ),
        # input_filter="$[1:]",
        # join_source="Input",
        # output_filter="$[0,-1]",
        # content_type="text/json",
        split_type="Line",
    )

    step_transform = TransformStep(
        name="OutOfStockTransform",
        step_args=step_args,
    )
    step_transform.add_depends_on([step_train])

    # Evaluate step for evaluating Model
    script_eval = ScriptProcessor(
        image_uri=custom_image_uri,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{base_job_prefix}/script-OutOfStock-eval",
        sagemaker_session=pipeline_session,
        role=role,
    )
    step_args = script_eval.run(
        inputs=[
            ProcessingInput(source=config_uri, destination="/opt/ml/processing/config"),
            ProcessingInput(
                source=f'{base_dir}/TEST/test.json',
                destination="/opt/ml/processing/test/",
            ),
            ProcessingInput(
                source=step_transform.properties.TransformOutput.S3OutputPath,
                destination="/opt/ml/processing/ouput/",
            ),
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation"),
        ],
        code=os.path.join(BASE_DIR, "evaluate.py"),
    )

    evaluation_report = PropertyFile(
        name="OutOfStockEvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )
    step_eval = ProcessingStep(
        name="EvaluateOutOfStockModel",
        step_args=step_args,
        property_files=[evaluation_report],
    )
    step_eval.add_depends_on([step_transform])

    # register model step that will be conditionally executed
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(
                step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
            ),
            content_type="application/json"
        )
    )

    # model_metrics = step_train.properties.FinalMetricDataList['test:RMSE'].Value
    model = Model(
        image_uri=image_uri,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        sagemaker_session=pipeline_session,
        role=role,
    )

    step_args = model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
    )

    step_register = ModelStep(
        name="RegisteOutOfStockModel",
        step_args=step_args,
    )

    # condition step for evaluating model quality and branching execution
    cond_lte = ConditionLessThanOrEqualTo(
        # left = step_train.properties.FinalMetricDataList['test:RMSE'].Value,
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="regression_metrics.rmse.value"
        ),
        right=8.0,
    )
    step_cond = ConditionStep(
        name="CheckMSOutOfStockEvaluation",
        conditions=[cond_lte],
        if_steps=[step_register],
        else_steps=[],
    )

    # pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            training_instance_type,
            model_approval_status,
            input_data,

        ],
        steps=[step_process, step_train, step_create_model, step_transform, step_eval, step_cond],
        sagemaker_session=pipeline_session,
    )
    return pipeline
