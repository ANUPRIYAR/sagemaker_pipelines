import json
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from datetime import datetime
import boto3
import os
import logging
import configparser

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

config = configparser.ConfigParser()
config.read('/opt/ml/processing/config/configurations.ini')

s3_bucket = config.get('OutOfStock_Properties', 's3_bucket')
s3_prefix = config.get('OutOfStock_Properties', 's3_prefix')

baseline_s3_prefix = config.get('OutOfStock_DriftCheck_Properties', 'baseline_s3_prefix')
reports_prefix = config.get('OutOfStock_DriftCheck_Properties', 'reports_prefix')

baseline_s3_uri = f's3://{s3_bucket}/{s3_prefix}/{baseline_s3_prefix}/'
reports_path = f's3://{s3_bucket}/{s3_prefix}/{reports_prefix}/'

register_new_model_baseline = eval(os.environ.get('register_new_model_baseline'))
logger.info(f"Input Parameter 'register_new_model_baseline' = {register_new_model_baseline}\n")

fail_job_flag = False

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + 0.001))) 


def join_gt_with_predictions(transform_output, ground_truth):
    # create next continous 28 dates 
    import pandas as pd
    start_date = ground_truth[0]['start']
    dt_list = pd.date_range(start=start_date, periods=28, freq='D')

    # slicing the groundtruth to length 28 targets
    for i in range(len(ground_truth) - 1):
        ground_truth[i]['target'] = ground_truth[i]['target'][:28]
        ground_truth[i]['start'] = dt_list
        cat = [ground_truth[i]['cat']] * 28
        ground_truth[i]['cat'] = cat
        try:
            ground_truth[i]['predictions'] = transform_output[i]['quantiles']['0.5']
        except Exception as e:
            logger.error(transform_output[i]['error'])
            raise

            # crop length of each array to the size of target
    for i in range(len(ground_truth) - 1):
        target_len = len(ground_truth[i]['target'])
        ground_truth[i]['start'] = ground_truth[i]['start'][:target_len]
        ground_truth[i]['cat'] = ground_truth[i]['cat'][:target_len]
        ground_truth[i]['predictions'] = ground_truth[i]['predictions'][:target_len]

    df = pd.DataFrame(columns=['start', 'target', 'cat'])
    for i in range(len(ground_truth) - 1):
        df_ = pd.DataFrame(ground_truth[i])
        df = pd.concat([df, df_])

    return df


def create_metrics_dict(gt_df):
    item_count = gt_df.shape[0]
    evaluation_time = datetime.now()
    # mae = mean_absolute_error(gt_df['target'].values, gt_df['predictions'].values)
    mape = mean_absolute_percentage_error(gt_df['target'].values, gt_df['predictions'].values)
    mse = mean_squared_error(gt_df['target'].values, gt_df['predictions'].values)
    rmse = np.sqrt(mse)
    # mape = mean_absolute_percentage_error(gt_df['target'].values, gt_df['predictions'].values)
    r2 = r2_score(gt_df['target'].values, gt_df['predictions'].values)

    # create metric dict 
    cur_metric_dict = {}
    cur_metric_dict['item_count'] = item_count
    cur_metric_dict['evaluation_time'] = evaluation_time
    cur_metric_dict['mse'] = mse
    cur_metric_dict['rmse'] = rmse
    cur_metric_dict['r2'] = r2
    cur_metric_dict['mape'] = mape
    return cur_metric_dict


def create_statistics_report_file(cur_metric_dict, s3_path):
    metrics_dict = {}
    metrics_dict['version'] = 0.0
    metrics_dict['dataset'] = {"item_count": cur_metric_dict['item_count'],
                               "evaluation_time": cur_metric_dict['evaluation_time']}
    metrics_dict['timeseries_forecast_metrics'] = {
        "mape": {
            "value": cur_metric_dict['mape']
            # "standard_deviation" : 0.0029995715409093385
        },
        "mse": {
            "value": cur_metric_dict['mse']
            # "standard_deviation" : 0.02085726710042106
        },
        "rmse": {
            "value": cur_metric_dict['rmse']
            # "standard_deviation" : 0.014754977908074233
        },
        "r2": {
            "value": cur_metric_dict['r2']
            # "standard_deviation" : 3.4628063400748763E-5
        }
    }

    json_stats = json.dumps(metrics_dict, indent=4, default=str)
    file_name = "statistics.json"
    with open(file_name, "w") as outfile:
        outfile.write(json_stats)

    # upload to s3 bucket
    s3_upload_path = s3_upload_file(s3_path, file_name)
    return s3_upload_path


def create_constraints_report_file(cur_metric_dict, s3_path):
    constraints_dict = {}
    constraints_dict['version'] = 0.0
    constraints_dict['timeseries_forecast_metrics'] = {
        "mape": {
            "threshold": cur_metric_dict['mape'],
            "comparison_operator": "GreaterThanThreshold"
        },
        "mse": {
            "threshold": cur_metric_dict['mse'],
            "comparison_operator": "GreaterThanThreshold"
        },
        "rmse": {
            "threshold": cur_metric_dict['rmse'],
            "comparison_operator": "GreaterThanThreshold"
        },
        "r2": {
            "threshold": cur_metric_dict['r2'],
            "comparison_operator": "LessThanThreshold"
        }
    }

    json_constraints = json.dumps(constraints_dict, indent=4, default=str)
    file_name = "constraints.json"
    with open(file_name, "w") as outfile:
        outfile.write(json_constraints)

    # upload to s3 bucket
    s3_upload_path = s3_upload_file(s3_path, file_name)
    return s3_upload_path


def s3_upload_file(s3_path, local_file):
    # Uploading the validation file to S3 bucket
    bucket_name = s3_path.split('/')[2]
    key = s3_path.split(bucket_name + '/')[1]
    boto3.Session().resource("s3").Bucket(bucket_name).Object(os.path.join(key, local_file)).upload_file(local_file)
    upload_path = s3_path + local_file
    return upload_path


def create_baseline(cur_metric_dict, baseline_s3_uri, register_new_model_baseline):
    if register_new_model_baseline:
        logger.info("Creating Model Quality Baseline...")
        stats_path = create_statistics_report_file(cur_metric_dict, baseline_s3_uri)
        constraints_path = create_constraints_report_file(cur_metric_dict, baseline_s3_uri)
        constraints_violations_path = generate_report(cur_metric_dict, baseline_s3_uri, 
                                                      register_new_model_baseline=register_new_model_baseline)
    return stats_path, constraints_path, constraints_violations_path


def create_reports(cur_metric_dict, baseline_dict, reports_path, register_new_model_baseline=False):
    if register_new_model_baseline == False:
        logger.info("Creating Model quality Reports...")
        stats_path = create_statistics_report_file(cur_metric_dict, reports_path)
        constraints_path = create_constraints_report_file(cur_metric_dict, reports_path)
        constraints_violations_path, fail_job_flag = generate_report(cur_metric_dict, reports_path, baseline_dict)

        return stats_path, constraints_path, constraints_violations_path, fail_job_flag


def read_baseline(source):
    # Code to download the file from s3 bucket
    session = boto3.Session()
    s3 = session.client('s3')
    bucket_name = source.split('/')[2]
    key = source.split(bucket_name + '/')[1]
    # print(f"bucket_name :{bucket_name} , key :{key}")
    s3_object = s3.get_object(Bucket=bucket_name, Key=key)
    body = s3_object['Body']
    data = body.read()
    baseline_dict = json.loads(data)
    return baseline_dict


def get_threshold_operator(baseline_dict, metric_name):
    threshold = baseline_dict['timeseries_forecast_metrics'][metric_name]['threshold']
    operator = baseline_dict['timeseries_forecast_metrics'][metric_name]['comparison_operator']
    return threshold, operator


def generate_report(cur_metric_dict, reports_path, baseline_dict=None, register_new_model_baseline=False):
    violation_report = {}
    fail_job_flag = False
    if register_new_model_baseline:
        violation_report["violations"] = []
    else:    
        report_list = []
        for metric_name in baseline_dict['timeseries_forecast_metrics']:
            metric_value = cur_metric_dict[metric_name]
            threshold, operator = get_threshold_operator(baseline_dict, metric_name)
            if operator == "GreaterThanThreshold":
                if float(metric_value) > float(threshold):
                    report_dict = {}
                    report_dict['constraint_check_type'] = operator
                    report_dict["description"] = f"Metric {metric_name} with {metric_value} was {operator} '{threshold}'"
                    report_dict["metric_name"] = metric_name
                    report_list.append(report_dict)
            elif operator == "LessThanThreshold":
                if float(metric_value) < float(threshold):
                    report_dict = {}
                    report_dict['constraint_check_type'] = operator
                    report_dict["description"] = f"Metric {metric_name} with {metric_value} was {operator} '{threshold}'"
                    report_dict["metric_name"] = metric_name
                    report_list.append(report_dict)
        if len(report_list) != 0:
            fail_job_flag = True
        violation_report["violations"] = report_list   
    constraints_violations_json = json.dumps(violation_report, indent=4, default=str)
    file_name = "constraints_violations.json"
    with open(file_name, "w") as outfile:
        outfile.write(constraints_violations_json)

    # upload to s3 bucket
    s3_upload_path = s3_upload_file(reports_path, file_name)
    return s3_upload_path, fail_job_flag


if __name__ == "__main__":
    base_dir = "/opt/ml/processing/"
    Ground_Truth_path = f"{base_dir}/TransformInput/oos_gt_data.json"
    Prediction_output_path = f"{base_dir}/TransformOutput/oos_gt_data.json.out"

    logger.info("Reading transform output and Ground Truth...")
    transform_output = []
    for line in open(Prediction_output_path, 'r'):
        transform_output.append(json.loads(line))

    ground_truth = []
    for line in open(Ground_Truth_path, 'r'):
        ground_truth.append(json.loads(line))

    logger.info(f"Transform_output : {transform_output}")
    logger.info(f"Ground Truth  : {ground_truth}")

    gt_df = join_gt_with_predictions(transform_output, ground_truth)
    logger.info(f"Shape of Ground Truth DataFrame : {gt_df.shape}")

    cur_metric_dict = create_metrics_dict(gt_df)
    logger.info(f"Current derived Metrics: {cur_metric_dict}")


    if register_new_model_baseline:
        stats_path, constraints_path, constraints_violations_path = create_baseline(cur_metric_dict, baseline_s3_uri,
                                                                                    register_new_model_baseline=register_new_model_baseline)
    else:
        source = baseline_s3_uri + "constraints.json"
        baseline_dict = read_baseline(source)
        logger.info(f"Baseline Metrics: {baseline_dict}")
        stats_path, constraints_path, constraints_violations_path, fail_job_flag = create_reports(cur_metric_dict, baseline_dict, reports_path, register_new_model_baseline=register_new_model_baseline)

    logger.info(f"Statistics Reports path : {stats_path}")
    logger.info(f"Constraints Reports path: {constraints_path}")
    logger.info(f"Constraints Violations Reports path: {constraints_violations_path}")
    
    if fail_job_flag:
        raise Exception("Baseline model constraints have been violated.")
