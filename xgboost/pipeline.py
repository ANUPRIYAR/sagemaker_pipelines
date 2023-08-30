"""
                                               . -ModelStep
                                              .
    Process-> Train -> Evaluate -> Condition .
                                              .
                                               . -(stop)

Implements a get_pipeline(**kwargs) method.
"""

from pipelines._utils import read_config
import os
import configparser

import boto3
import sagemaker
import sagemaker.session

from sagemaker.estimator import Estimator
from sagemaker.xgboost import XGBoostPredictor
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
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
    ParameterBoolean
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
    TuningStep,
    CacheConfig,
     TransformStep
    
)
from sagemaker.workflow.model_step import ModelStep
from sagemaker.model import Model
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.tuner import (
    IntegerParameter,
    CategoricalParameter,
    ContinuousParameter,
    HyperparameterTuner,
     WarmStartConfig,
    WarmStartTypes,
)
from sagemaker.transformer import Transformer
from sagemaker.inputs import BatchDataCaptureConfig
from sagemaker.inputs import TrainingInput, CreateModelInput, TransformInput
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
from sagemaker.model_monitor import DatasetFormat, model_monitoring
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.check_job_config import CheckJobConfig


BASE_DIR = os.path.dirname(os.path.realpath(__file__))
print(BASE_DIR)

# Configuration File
config_bucket = 'pipeline-aiml-stg'
# config_key = 'dynamic_campaign_dataset/Configurations/configurations.ini'
config_key = 'BCM_Pipeline_Data/Configurations/configurations.ini'
config_uri = f"s3://{config_bucket}/{config_key}"

# Initialize the Parser object
config = configparser.ConfigParser()
config_string = read_config(bucket=config_bucket ,key=config_key)
config.read_string(config_string.decode("utf-8"))

# Reading the configuration variables
bucket = config.get('BCM_Properties', 's3_bucket')
key = config.get('BCM_Properties', 'key')
s3_prefix = config.get('BCM_Properties', 's3_prefix')
data_capture_prefix = config.get('BCM_Properties', 'data_capture_prefix')
baseline_results_prefix = config.get('BCM_Properties', 'baseline_results_prefix')

# baseline_data_results_prefix = config.get('BCM_Properties', 'baseline_data_results_prefix')
s3_report_path_prefix = config.get('BCM_Properties', 's3_report_path_prefix')
baseline_data_results_prefix = config.get('BCM_Properties', 'baseline_data_results_prefix')

data_baseline_path = f"s3://{bucket}/{baseline_data_results_prefix}"
# baseline_data_results_prefix = 'BcmPipeline/DataMonitor/baselining/results/'
print(f"Capture prefix : {data_capture_prefix}")
print(f"Baseline results prefix : {baseline_results_prefix}")
print(f"s3_report_path_prefix : {s3_report_path_prefix}")
print(f"Data Baseline results path : {data_baseline_path}")


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

def get_pipeline_custom_tags(new_tags, region, sagemaker_project_name=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.describe_project(ProjectName=sagemaker_project_name)
        sagemaker_project_arn = response["ProjectArn"]
        response = sm_client.list_tags(
            ResourceArn=sagemaker_project_arn)
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags


def get_pipeline(
    region,
    sagemaker_project_name=None,
    role=None,
    default_bucket=None,
    model_package_group_name="BcmPackageGroup",
    pipeline_name="BcmPipeline",
    base_job_prefix="BCM",
    processing_instance_type="ml.m5.xlarge",
    training_instance_type="ml.m5.xlarge",
):
    """Gets a SageMaker ML Pipeline instance working with on abalone data.

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
    
    # Cache Pipeline steps to reduce execution time on subsequent executions
    cache_config = CacheConfig(enable_caching=True, expire_after="30d")
    input_data = ParameterString(
        name="InputDataUrl",
        default_value=f"s3://{bucket}/{key}",
    )
    

    
    # processing step for feature engineering
    sklearn_processor = SKLearnProcessor(
        framework_version="0.23-1",
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}/sklearn-bcm-preprocess",
        sagemaker_session=pipeline_session,
        role=role,
    )
    step_args = sklearn_processor.run(
        inputs=[
            ProcessingInput(source=config_uri, destination="/opt/ml/processing/config"),
            ProcessingInput(source=input_data, destination="/opt/ml/processing/input_data"),
        ],
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
        ],
        code=os.path.join(BASE_DIR, "preprocess.py"),
        arguments=["--input-data", input_data],
    )
    step_process = ProcessingStep(
        name="PreprocessBCMData",
        step_args=step_args,
    )
    
    check_job_config = CheckJobConfig(
    role=role,
    instance_count=1,
    instance_type="ml.c5.xlarge",
    volume_size_in_gb=120,
    sagemaker_session=pipeline_session,
    )
     
    
    data_quality_check_config = DataQualityCheckConfig(
        baseline_dataset=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
        dataset_format=DatasetFormat.csv(header=True, output_columns_position="START"), 
        output_s3_uri = Join(on='/', values=['s3:/', bucket, baseline_data_results_prefix ])
        
    )

    data_quality_check_step = QualityCheckStep(
        name="DataQualityCheckStep",
        skip_check=True, 
        register_new_baseline = True,
        quality_check_config=data_quality_check_config,
        check_job_config=check_job_config,
        supplied_baseline_statistics = f"{data_baseline_path}/statistics.json",
        supplied_baseline_constraints = f"{data_baseline_path}/constraints.json",
        model_package_group_name=model_package_group_name
    )

    sklearn_processor = SKLearnProcessor(
        framework_version="0.23-1",
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}/update-constraints",
        sagemaker_session=pipeline_session,
        role=role,
    )
    
    step_args = sklearn_processor.run(
        inputs=[
            ProcessingInput(source=config_uri, destination="/opt/ml/processing/config"),
            ProcessingInput(source=f"{data_baseline_path}/constraints.json", destination="/opt/ml/processing/constraints/"),
        ],
        code=os.path.join(BASE_DIR, "update_constraints.py"),
       
    )
    step_update = ProcessingStep(
        name="UpdateConstraints",
        step_args=step_args,
        depends_on=[data_quality_check_step]
    )
    
    # training step for generating model artifacts
    # step_tuning 
    model_path = f"s3://{sagemaker_session.default_bucket()}/{base_job_prefix}/BCMTrain"
    image_uri = sagemaker.image_uris.retrieve(
        framework="xgboost",
        region=region,
        version="1.0-1",
        py_version="py3",
        instance_type=training_instance_type,
    )
    xgb_train = Estimator(
        image_uri=image_uri,
        instance_type=training_instance_type,
        instance_count=1,
        output_path=model_path,
        base_job_name=f"{base_job_prefix}/bcm-train",
        sagemaker_session=pipeline_session,
        role=role,
    )
    xgb_train.set_hyperparameters(
        # objective="reg:linear",
        num_round=150,
        # max_depth=5,
        # eta=0.2,
        gamma=1.744,
        # min_child_weight=6,
        subsample=0.61,
        silent=0,
    )
    
    objective_metric_name = "validation:rmse"
    
    hyperparameter_ranges = {
         'alpha': ContinuousParameter(0.01, .2),
        'eta': ContinuousParameter(0.3, .5),
        'min_child_weight': ContinuousParameter(1., 2.),
        'max_depth': IntegerParameter(8, 10),
         'gamma': ContinuousParameter(1,2),
        'colsample_bytree': ContinuousParameter(0.5, 1),
        # 'subsample': ContinuousParameter(0.5, 1)
    }
    
    tuner_log = HyperparameterTuner(
        xgb_train,
        objective_metric_name,
        hyperparameter_ranges,
        max_jobs=5,
        max_parallel_jobs=5,
        strategy="Bayesian",
        objective_type="Minimize",
    )   
                
    hpo_args = tuner_log.fit(
        inputs={
            "train": TrainingInput(
          s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                        content_type="text/csv",
                    ),
                    "validation": TrainingInput(
                        s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                            "validation"
                        ].S3Output.S3Uri,
                        content_type="text/csv",
                    ),
                }
            )

    step_tuning = TuningStep(
        name="HPTuning",
        step_args=hpo_args,
        cache_config=cache_config,
    )
    
    #   step_tuning_warm_start  
    parent_tuning_job_name = (
    step_tuning.properties.HyperParameterTuningJobName
    )
    
    
   #   step_tuning_warm_start  
    parent_tuning_job_name = (
    step_tuning.properties.HyperParameterTuningJobName
    )  # Use the parent tuning job specific to the use case

    warm_start_config = WarmStartConfig(
        WarmStartTypes.IDENTICAL_DATA_AND_ALGORITHM, parents={parent_tuning_job_name}
    )
    

    tuner_log_warm_start = HyperparameterTuner(
        xgb_train,
        objective_metric_name,
        hyperparameter_ranges,
        max_jobs=3,
        max_parallel_jobs=5,
        strategy="Bayesian",
        objective_type="Minimize",
        warm_start_config=warm_start_config,
    )

    tuner_run_args = tuner_log_warm_start.fit(
        inputs={
            "train": TrainingInput(s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "validation"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
        }
    )

    step_tuning_warm_start = TuningStep(
        name="HPTuningWarmStart",
        step_args=tuner_run_args,
        cache_config=cache_config,
    )
    
    
    
    #  step_create_first to choose the best model 
    model_prefix = f"{base_job_prefix}/BCMTrain"

    best_model = Model(
        image_uri=image_uri,
        model_data=step_tuning.get_top_model_s3_uri(
            top_k=0, s3_bucket=default_bucket, prefix=model_prefix
        ),
        predictor_cls=XGBoostPredictor,
        sagemaker_session=pipeline_session,
        role=role,
    )

    step_create_first = ModelStep(
        name="CreateBestModel",
        step_args=best_model.create(instance_type="ml.m5.xlarge"),
    )
    
    #  Transform step
    transformer = Transformer(
        model_name=step_create_first.properties.ModelName,
        instance_type="ml.m5.xlarge",
        instance_count=1,
        accept="text/csv",
        assemble_with="Line",
        output_path=f"s3://{bucket}/{s3_prefix}/BCMTransform/",
        sagemaker_session=pipeline_session,
    )

    # The output of the transform step combines the prediction and the input label.
    # The output format is `prediction, original label`

    transform_inputs = TransformInput(
        data=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
    )

    step_args = transformer.transform(
        data=transform_inputs.data,
        input_filter="$[1:]", 
        join_source="Input",
        output_filter="$[0,-1]",
        content_type="text/csv",
        split_type="Line",
        #  Data capture
        batch_data_capture_config=BatchDataCaptureConfig(
        destination_s3_uri = f"s3://{bucket}/{data_capture_prefix}",
        # kms_key_id="kms_key",
        generate_inference_id=True,)
    )

    step_transform = TransformStep(
        name="BCMTransform",
        step_args=step_args,
    )
    
    

   # Model Quality check
    model_quality_check_config = ModelQualityCheckConfig(
        baseline_dataset=step_transform.properties.TransformOutput.S3OutputPath,
        dataset_format=DatasetFormat.csv(header=False),
        output_s3_uri=Join(on='/', values=['s3:/', bucket, baseline_results_prefix]),
        problem_type='Regression',
        inference_attribute='_c1',
        ground_truth_attribute='_c0',
    )

    model_quality_check_step = QualityCheckStep(
        name="ModelQualityCheckStep",
        skip_check=True,
        register_new_baseline = True,
        quality_check_config=model_quality_check_config,
        check_job_config=check_job_config,
        supplied_baseline_statistics=f"s3://{bucket}/{baseline_results_prefix}/statistics.json",
        supplied_baseline_constraints=f"s3://{bucket}/{baseline_results_prefix}/constraints.json",
        model_package_group_name=model_package_group_name,
    )
      

    
    # processing step for evaluation
    script_eval = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{base_job_prefix}/script-bcm-eval",
        sagemaker_session=pipeline_session,
        role=role,
    )
    step_args = script_eval.run(
        inputs=[
            ProcessingInput(source=config_uri, destination="/opt/ml/processing/config"),
            ProcessingInput(
                source = step_tuning.get_top_model_s3_uri(
                top_k=0, s3_bucket=default_bucket, prefix=model_prefix
            ),
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs[
                    "test"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ), 
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation"),
        ],
        code=os.path.join(BASE_DIR, "evaluate.py"),
    )
    
    evaluation_report = PropertyFile(
        name="BcmEvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )
    
    step_eval = ProcessingStep(
        name="EvaluateBcmModel",
        step_args=step_args,
        property_files=[evaluation_report],
    )


    model_metrics = ModelMetrics(
         model_data_statistics=MetricsSource(
            s3_uri=data_quality_check_step.properties.CalculatedBaselineStatistics,
            content_type="application/json",
        ),
        model_data_constraints=MetricsSource(
            s3_uri=data_quality_check_step.properties.CalculatedBaselineConstraints,
            content_type="application/json",
        ),
         model_statistics=MetricsSource(
            s3_uri=model_quality_check_step.properties.CalculatedBaselineStatistics,
            content_type="application/json",
        ),
        model_constraints=MetricsSource(
            s3_uri=model_quality_check_step.properties.CalculatedBaselineConstraints,
            content_type="application/json",
        ),
    )
        
    drift_check_baselines = DriftCheckBaselines(
        model_data_statistics=MetricsSource(
            s3_uri=data_quality_check_step.properties.BaselineUsedForDriftCheckStatistics,
            content_type="application/json",
        ),
        model_data_constraints=MetricsSource(
            s3_uri=data_quality_check_step.properties.BaselineUsedForDriftCheckConstraints,
            content_type="application/json",
        ),
        model_statistics=MetricsSource(
            s3_uri=model_quality_check_step.properties.BaselineUsedForDriftCheckStatistics,
            content_type="application/json",
        ),
        model_constraints=MetricsSource(
            s3_uri=model_quality_check_step.properties.BaselineUsedForDriftCheckConstraints,
            content_type="application/json",
        ),
    )
            
    

    model = Model(
        image_uri=image_uri,
        model_data = step_tuning.get_top_model_s3_uri(
                top_k=0, s3_bucket=default_bucket, prefix=model_prefix
            ),
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
        name="RegisterBcmModel",
        step_args=step_args,
    )

    # condition step for evaluating model quality and branching execution
    cond_lte = ConditionLessThanOrEqualTo(
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="regression_metrics.rmse.value"
        ),
        right=12.0,
    )
    step_cond = ConditionStep(
        name="CheckMSEBcmEvaluation",
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
        steps=[step_process,data_quality_check_step, step_update, step_tuning, step_create_first, step_transform, model_quality_check_step, step_eval, step_cond]
       
    )
    return pipeline
