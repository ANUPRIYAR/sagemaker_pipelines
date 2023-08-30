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
from sagemaker.sklearn.estimator import SKLearn

from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput,  CreateModelInput, TransformInput
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
from sagemaker.workflow.check_job_config import CheckJobConfig
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



# BASE_DIR = os.path.dirname(os.path.realpath(__file__))
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
s3_bucket = 'sagemaker-eu-west-1-610914939903'
base_dir = 's3://sagemaker-eu-west-1-610914939903/oos_prediction/OutOfStock/'
s3_prefix = "oos_prediction"
output_folder = 'ouput'
data_capture_destination = f'{base_dir}DataCapture/'
print(data_capture_destination)
# Set up hyperparameters and training job configuration
freq = 'D'
prediction_length = 28
# epochs = 219
epochs = 1
learning_rate = 0.001
num_cells = 40
num_layers = 2
context_length = 28
# cat=True
cardinality="auto"
num_dynamic_feat="ignore"

# Uri
custom_uri = '610914939903.dkr.ecr.eu-west-1.amazonaws.com/forecasting-processing:latest'

baseline_results_uri = "s3://sagemaker-eu-west-1-610914939903/sagemaker/DEMO-ModelMonitor/baselining/results"
supplied_baseline_statistics_model_quality = f"{baseline_results_uri}/statistics.json"
supplied_baseline_constraints_model_quality = f"{baseline_results_uri}/constraints.json"

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
    model_package_group_name="OutOfStockForecastPackageGroup",
    pipeline_name="OutOfStockForecastPipeline",
    base_job_prefix="OutOfStockForecast",
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
    # input_data = ParameterString(
    #     name="InputDataUrl",
    #     default_value="s3://sagemaker-eu-west-1-610914939903/oos_prediction/data/vc_forecast_and_inventory_ordered_units.csv",
    # )
    
    # for model quality check step
    skip_check_model_quality = ParameterBoolean(name="SkipModelQualityCheck", default_value = False)
    register_new_baseline_model_quality = ParameterBoolean(name="RegisterNewModelQualityBaseline", default_value=False)
    supplied_baseline_statistics_model_quality = ParameterString(name="ModelQualitySuppliedStatistics", default_value='')
    supplied_baseline_constraints_model_quality = ParameterString(name="ModelQualitySuppliedConstraints", default_value='')

    # processing step for feature engineering
    input_train = ParameterString(
        # name="TrainData",  
       name = "input_train", 
       default_value = f"s3://{default_bucket}/oos_prediction/data/data.csv",
        #default_value=f"s3://{default_bucket}/oos_prediction/data/Forecast/Formatted_Data/Syn_Timeseries_2013_2023.csv"
    )
    
    model_output = ParameterString(name="ModelOutput", default_value=f"s3://{default_bucket}/model")

    # Model parameters
    forecast_horizon = ParameterString(
        name="ForecastHorizon", default_value="24"
    )
    forecast_algorithm = ParameterString(
        name="ForecastAlgorithm", default_value="NPTS"
    )
    maximum_score = ParameterString(
        name="MaxScore", default_value="0.4"
    )
    metric = ParameterString(
        name="EvaluationMetric", default_value="WAPE"
    )

    image_uri = sagemaker.image_uris.retrieve(
        framework="sklearn",  # we are using the Sagemaker built in xgboost algorithm
        region=region,
        version="1.0-1",
        py_version="py3",
        instance_type="ml.c5.xlarge",
    )
    
    sklearn_processor = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}/forecast-process",
        sagemaker_session=pipeline_session,
        role=role,
    )
    
    step_process = ProcessingStep(
        name="ForecastPreProcess",
        processor=sklearn_processor,
        inputs=[
            ProcessingInput(source=input_train, destination="/opt/ml/processing/input_train"),
        ],
        outputs=[
            ProcessingOutput(output_name="target", source="/opt/ml/processing/target"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
            # ProcessingOutput(output_name="related", source="/opt/ml/processing/related"),
        ],
        job_arguments=["--forecast_horizon", forecast_horizon],
        code=os.path.join(BASE_DIR, "preprocess.py"),
    )

    
    
#     check_job_config = CheckJobConfig(
#     role=role,
#     instance_count=1,
#     instance_type="ml.c5.xlarge",
#     volume_size_in_gb=120,
#     sagemaker_session=pipeline_session,
# )


    # training step for generating model artifacts
    # Define the hyperparmeters and the Regex Syntax associated to the metrics
    hyperparameters = {
        "forecast_horizon": forecast_horizon,
        "forecast_algorithm": forecast_algorithm,
        "dataset_frequency": "D",
        "timestamp_format": "yyyy-MM-dd",
        "number_of_backtest_windows": "1",
        "s3_directory_target": step_process.properties.ProcessingOutputConfig.Outputs[
            "target"
        ].S3Output.S3Uri,
        # "s3_directory_related": preprocess.properties.ProcessingOutputConfig.Outputs[
        #     "related"
        # ].S3Output.S3Uri,
        "role_arn": role,
        "region": region,
    }
    metric_definitions = [
        {"Name": "WAPE", "Regex": "WAPE=(.*?);"},
        {"Name": "RMSE", "Regex": "RMSE=(.*?);"},
        {"Name": "MASE", "Regex": "MASE=(.*?);"},
        {"Name": "MAPE", "Regex": "MAPE=(.*?);"},
    ]

    forecast_model = SKLearn(
        entry_point=os.path.join(BASE_DIR, "train.py"),
        role=role,
        image_uri=image_uri,
        #instance_count=training_instance_count,
        instance_type=training_instance_type,
        sagemaker_session=pipeline_session,
        base_job_name=f"{base_job_prefix}/forecast-train",
        hyperparameters=hyperparameters,
        enable_sagemaker_metrics=True,
        metric_definitions=metric_definitions,
    )
    

    forecast_train_and_eval = TrainingStep(
        name="ForecastTrainAndEvaluate", estimator=forecast_model
    )
    
    forecast_train_and_eval.add_depends_on([step_process])
    
    
    
    # Create Model step
    model = Model(
        image_uri=image_uri,
        model_data=forecast_train_and_eval.properties.ModelArtifacts.S3ModelArtifacts,
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
    # environment_param = {
    #     'num_samples': 20,
    #     'output_types': ['quantiles'],
    #     'quantiles': ['0.5']
    # }
    
#     transformer = Transformer(
#     # execution_input["ModelName"],
#     model_name=step_create_model.properties.ModelName,
#     instance_type="ml.m5.xlarge",
#     instance_count=1,
#     # output_path=f's3://{s3_bucket}/{s3_prefix}/OutOfStock/Forecast/BATCH_TRANSFORM/OUTPUT/',
#     output_path = f"s3://{s3_bucket}/{s3_prefix}/OutOfStock/FORECAST/Batch_Transform/output/",
#     sagemaker_session=pipeline_session,
#     accept = 'application/csv',
#     strategy='MultiRecord',
#     assemble_with='Line',
#     # env = {
#     #     'DEEPAR_INFERENCE_CONFIG': json.dumps(environment_param)
#     # },
       
    
#     )
    
#     transform_inputs = TransformInput(
#         data=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
#         # data = f's3://{s3_bucket}/{s3_prefix}/OutOfStock/BATCH_TRANSFORM/INPUT/batch_transform_input.json'
#     )
    
#     step_args = transformer.transform(
#         data=transform_inputs.data,
#     #     batch_data_capture_config=BatchDataCaptureConfig(
#     #     destination_s3_uri=data_capture_destination,
#     #     # kms_key_id="kms_key",
#     #     generate_inference_id=True
#     # ),
#         # model_name = step_create_model.properties.ModelName,
#         input_filter="$[1:]",
#         # join_source="Input",
#         # output_filter="$[0,-1]",
#         content_type="text/csv",
#         split_type="Line",
#     )
    
#     step_transform = TransformStep(
#         name="OutOfStockTransform",
#         step_args=step_args,
#     )
    # step_transform.add_depends_on([step_train])
    
    
#     # Model Quality check
#     model_quality_check_config = ModelQualityCheckConfig(
#         baseline_dataset=step_transform.properties.TransformOutput.S3OutputPath,
#         dataset_format=DatasetFormat.csv(header=False),
#         output_s3_uri=Join(on='/', values=['s3:/', default_bucket, base_job_prefix, ExecutionVariables.PIPELINE_EXECUTION_ID, 'modelqualitycheckstep']),
#         problem_type='Regression',
#         inference_attribute='_c0',
#         ground_truth_attribute='_c1'
#     )

#     model_quality_check_step = QualityCheckStep(
#         name="ModelQualityCheckStep",
#         skip_check=skip_check_model_quality,
#         register_new_baseline=register_new_baseline_model_quality,
#         quality_check_config=model_quality_check_config,
#         check_job_config=check_job_config,
#         supplied_baseline_statistics=supplied_baseline_statistics_model_quality,
#         supplied_baseline_constraints=supplied_baseline_constraints_model_quality,
#         model_package_group_name=model_package_group_name
#     )

    
    
    
    
    # Evaluate step for evaluating Model
    script_eval = ScriptProcessor(
        image_uri=custom_uri,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{base_job_prefix}/script-OutOfStock-eval",
        sagemaker_session=pipeline_session,
        role=role,
    )
    step_args = script_eval.run(
        inputs=[
            ProcessingInput(
                source=forecast_train_and_eval.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            # ProcessingInput(
            #     source=step_process.properties.ProcessingOutputConfig.Outputs[
            #         "test"
            #     ].S3Output.S3Uri,
            #     destination="/opt/ml/processing/test",
            # ),
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
#     step_eval.add_depends_on([step_transform])
    
    
    # register model step that will be conditionally executed
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(
                step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
            ),
            content_type="application/json"
        )
    
    #      model_statistics=MetricsSource(
    #         s3_uri=model_quality_check_step.properties.CalculatedBaselineStatistics,
    #         content_type="application/json",
    #     ),
    #     model_constraints=MetricsSource(
    #         s3_uri=model_quality_check_step.properties.CalculatedBaselineConstraints,
    #         content_type="application/json",
    #     ),
    )
    
#     drift_check_baselines = DriftCheckBaselines(
#         model_statistics=MetricsSource(
#             s3_uri=model_quality_check_step.properties.BaselineUsedForDriftCheckStatistics,
#             content_type="application/json",
#         ),
#         model_constraints=MetricsSource(
#             s3_uri=model_quality_check_step.properties.BaselineUsedForDriftCheckConstraints,
#             content_type="application/json",
#         ),
#     )
    
    
    
    # model_metrics = step_train.properties.FinalMetricDataList['test:RMSE'].Value
    model = Model(
        image_uri=image_uri,
        model_data=forecast_train_and_eval.properties.ModelArtifacts.S3ModelArtifacts,
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
        # drift_check_baselines=drift_check_baselines,
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
        right=30.0,
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
            # input_data,
            input_train,
            forecast_horizon,
            forecast_algorithm,
            model_output,
            metric
            # skip_check_model_quality,
            # register_new_baseline_model_quality,
            # supplied_baseline_statistics_model_quality,
            # supplied_baseline_constraints_model_quality,

        ],
        # steps=[step_process, step_train, step_create_model, step_transform,  step_eval, step_cond],
        steps=[step_process, forecast_train_and_eval, step_create_model, step_eval, step_cond],
        sagemaker_session=pipeline_session,
    )
    return pipeline
