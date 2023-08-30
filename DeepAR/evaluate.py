"""Evaluation script for measuring mean squared error."""
import json
import logging
import pathlib
import boto3
import pandas as pd
import numpy as np
import configparser
from sklearn.metrics import mean_squared_error

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

config = configparser.ConfigParser()
config.read('/opt/ml/processing/config/configurations.ini')

s3_bucket = config.get('OutOfStock_Properties', 's3_bucket')
s3_prefix = config.get('OutOfStock_Properties', 's3_prefix')
base_dir = f's3://{s3_bucket}/{s3_prefix}'



def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + 0.001))) * 100


def join_test_with_predictions(transform_output, ground_truth):
    # create next continous 28 dates 
    start_date = ground_truth[0]['start']
    dt_list = pd.date_range(start=start_date, periods=28, freq='D')

    # slicing the groundtruth to length 28 targets
    for i in range(len(ground_truth) - 1):
        ground_truth[i]['target'] = ground_truth[i]['target'][:28]
        ground_truth[i]['start'] = dt_list
        cat = [ground_truth[i]['cat']] * 28
        ground_truth[i]['cat'] = cat
        ground_truth[i]['predictions'] = transform_output[i]['quantiles']['0.5']

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
    mse = mean_squared_error(gt_df['target'].values, gt_df['predictions'].values)
    rmse = np.sqrt(mse)
    std = np.std(gt_df['target'].values - gt_df['predictions'].values)
    mape = mean_absolute_percentage_error(gt_df['target'].values, gt_df['predictions'].values)

    # create metric dict 
    cur_metric_dict = {}
    cur_metric_dict['rmse'] = rmse
    cur_metric_dict['std'] = std
    cur_metric_dict['mape'] = mape
    return cur_metric_dict


if __name__ == "__main__":
    logger.debug("Starting evaluation.")
    test_file_path = "/opt/ml/processing/test/test.json"
    batchtransform_output_file_path = "/opt/ml/processing/ouput/batch_transform_input.json.out"

    transform_output = []
    for line in open(batchtransform_output_file_path, 'r'):
        transform_output.append(json.loads(line))

    test = []
    for line in open(test_file_path, 'r'):
        test.append(json.loads(line))

    test_df = join_test_with_predictions(transform_output, test)
    logger.info(f"Shape of DataFrame : {test_df.shape}")

    cur_metric_dict = create_metrics_dict(test_df)
    logger.info(f"cur_metric_dict : {cur_metric_dict}")

    mape = cur_metric_dict['mape']
    rmse = cur_metric_dict['rmse']
    std = cur_metric_dict['std']

    report_dict = {
        "regression_metrics": {
            "rmse": {
                "value": rmse,
                "standard_deviation": std
                # "mean_absolute_percentage_error": mape
            },
        },
    }

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Writing out evaluation report with rmse: %f", rmse)
    logger.info(f"Calculated Metrics\n rmse: {rmse} \n mean_absolute_percentage_error : {mape}")
    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))

    s3_client = boto3.client("s3")
    s3_client.upload_file(evaluation_path, s3_bucket, s3_prefix + "/evaluation/" + evaluation_path)
    logger.info(f"Evaluation file written to: {evaluation_path}")
