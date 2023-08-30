import ast
import boto3
import pandas as pd
import logging
import configparser
import pathlib
import json
import csv

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

config = configparser.ConfigParser()
config.read('/opt/ml/processing/config/configurations.ini')

s3_bucket = config.get('OutOfStock_Properties', 's3_bucket')
s3_prefix = config.get('OutOfStock_Properties', 's3_prefix')
file_train = config.get('OutOfStock_Preprocess', 'file_train')
file_test = config.get('OutOfStock_Preprocess', 'file_test')
product_mapping_key = config.get('OutOfStock_Properties', 'product_mapping_key')

interpolation_technique = config.get('OutOfStock_Preprocess', 'interpolation_technique')
freq = config.get('OutOfStock_Preprocess', 'freq')
features = config.get('OutOfStock_Preprocess', 'features')
date_col = config.get('OutOfStock_Preprocess', 'date_col')
asin_col = config.get('OutOfStock_Preprocess', 'asin_col')
units_sold_col = config.get('OutOfStock_Preprocess', 'units_sold_col')
asin_cat_col = config.get('OutOfStock_Preprocess', 'asin_cat_col')
file_batch_transform = config.get('OutOfStock_Preprocess', 'file_batch_transform')
base_dir = f's3://{s3_bucket}/{s3_prefix}'

raw_data_prefix = config.get('OutOfStock_Properties', 'raw_data_prefix')
function_arn = config.get('OutOfStock_DriftCheck_Properties', 'function_arn')
lambda_region = config.get('OutOfStock_DriftCheck_Properties', 'lambda_region')


# trigger Lambda Function to fetch data from DB
def fetch_data_from_lambda_and_save_to_csv(s3_bucket, prefix):
    lambda_client = boto3.client('lambda', region_name=lambda_region)  # Create a Boto3 client for Lambda       
    
    # Invoke the Lambda function
    response = lambda_client.invoke(
        FunctionName=function_arn,
        InvocationType='RequestResponse'
    )
    payload = response['Payload'].read()  # Retrieve the response payload
    data = json.loads(payload)
    
    key = data['body']['path']  # Access the retrieved data
    print(key)
    return key


def upsampling(df, freq, interpolation_technique):
    grouped = df.groupby(asin_col)
    dfs = []
    for title, group in grouped:
        group.set_index(date_col, inplace=True)
        group = group.resample(freq).mean()  # Resample the data at daily frequency to fill in missing dates 
        # Interpolate the missing values using linear interpolation
        group[units_sold_col] = group[units_sold_col].interpolate(method=interpolation_technique)
        group[asin_col] = title  # Add the product_title column and reset the index
        group = group.reset_index()
        dfs.append(group)

    df_upsampled = pd.concat(dfs)
    df_upsampled[date_col] = df_upsampled[date_col].dt.strftime('%Y-%m-%d %H:%M:%S')
    # df_upsampled[date_col] = pd.to_datetime(df_upsampled[date_col])
    return df_upsampled


# Convert the asin to encoded asin from Product Mapping sheet 
def convert_to_encoded(df):
    s3 = boto3.resource('s3')
    key = product_mapping_key
    content_object = s3.Object(s3_bucket, key)
    file_content = content_object.get()['Body'].read().decode('utf-8')
    json_content = json.loads(file_content)
    df['asin_code'] = df['asin'].apply(lambda x: json_content.get(x, '100'))
    logger.info(f"Converted asins to encoding: {df.head(5)}")
    logger.info(f" Columns: {df.columns}")
    return df


def feature_engineering(df, freq, interpolation_technique):
    logger.info("Extracting features from raw data..")
    columns = ast.literal_eval(features)
    df = df[columns]
    df[date_col] = pd.to_datetime(df[date_col], utc=True)

    logger.info(f"Upsampling data with Frequency: {freq} \n interpolation : {interpolation_technique}")
    df_upsampled = upsampling(df, freq, interpolation_technique)
    # df_upsampled[date_col] = df_upsampled[date_col].dt.strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"Upsampling complete. Shape: {df_upsampled.shape}")

    #  Encode categorical Variable   
    logger.info("Converting the asins to its encoded values through Product Mapping Sheet")
    df_upsampled = convert_to_encoded(df_upsampled)

    # Drop the column asin_col
    df_upsampled.drop(asin_col, axis=1, inplace=True)
    df_upsampled.reset_index(inplace=True)
    return df_upsampled


# Creating Training and Test data in json format
def get_timeseries(df):
    product_ids = df[asin_cat_col].unique().tolist()
    df.set_index(date_col, inplace=True)
    start_vals = []
    timeseries = []
    cat_vals = []
    for pid in product_ids:
        cat_vals.append(int(pid))
        start_vals.append(df[df[asin_cat_col] == pid].index[0])
        timeseries.append(df[df[asin_cat_col] == pid][units_sold_col])
    return timeseries, start_vals, cat_vals


def series_to_obj(ts, cat=None, start=None):
    obj = {"start": start, "target": list(ts)}
    if cat is not None:
        obj["cat"] = cat
    return obj


def series_to_jsonline(ts, start, cat):
    return json.dumps(series_to_obj(ts, cat, start))


def write_to_bucket(File, s3_bucket, s3_prefix, timeseries, start_vals, cat_vals, s3_folder):
    encoding = "utf-8"
    with open(File, "wb") as f:
        for i, ts in enumerate(timeseries):
            f.write(series_to_jsonline(ts, str(start_vals[i]), cat_vals[i]).encode(encoding))
            f.write("\n".encode(encoding))

    file_name = "s3://" + s3_bucket + "/" + s3_prefix + "/" + s3_folder + "/" + File
    logger.info(f"Uploading json file.\n S3 Location: {file_name}")
    s3 = boto3.client("s3")
    s3.upload_file(File, s3_bucket, s3_prefix + "/" + s3_folder + "/" + File)
    return file_name


def get_batchtransform_input(timeseries_train, train_cat_vals, test_start_vals, test_cat_vals):
    prediction_series = []
    prediction_cat_values = []
    for cat in test_cat_vals:
        if cat in train_cat_vals:
            prediction_cat_values.append(cat)
            cat_index = train_cat_vals.index(cat)
            prediction_series.append(timeseries_train[cat_index][-28:])
    prediction_start_values = test_start_vals
    return prediction_series, prediction_start_values, prediction_cat_values


if __name__ == "__main__":
    prefix = s3_prefix + "/" + raw_data_prefix
    logger.info(f"Ground Truth prefix : {prefix}")
    # Fetch the Ground Truth data from DB
    logger.info("Calling the Lambda funtion to fetch the Ground Truth data from DB")
    Raw_GT_path = fetch_data_from_lambda_and_save_to_csv(s3_bucket, prefix)
    Raw_GT_key = Raw_GT_path.split(s3_bucket + '/')[1]
    logger.info(f" S3 Key for Raw Ground Truth data : {Raw_GT_key}")
    s3 = boto3.client('s3')
    obj = s3.get_object(
        Bucket=s3_bucket,
        Key=Raw_GT_key
    )
    raw_df = pd.read_csv(obj['Body'], parse_dates=True)
    
    df = raw_df.copy()
    logger.info(f"Shape : {df.shape}")

    logger.info("Feature Engineering started...")
    df_upsampled = feature_engineering(df, freq, interpolation_technique)
    logger.info(f"Feature Engineering completed.\n Dataset shape: {df_upsampled.shape}")

    # splitting the data into half to get previous 28 ordered units to be used as Batch Transform Input   
    half = int(df_upsampled.shape[0] / 2)
    div_date = df_upsampled['date'][half]

    train_data = df_upsampled[df_upsampled[date_col] < div_date]
    val_data = df_upsampled[df_upsampled[date_col] >= div_date]

    # Getting the timeseries train and test values to be used in creating Batch Transform Input    
    timeseries_train, train_start_vals, train_cat_vals = get_timeseries(train_data)
    timeseries_test, test_start_vals, test_cat_vals = get_timeseries(val_data)

    prediction_series, prediction_start_values, prediction_cat_values = get_batchtransform_input(timeseries_train,
                                                                                                 train_cat_vals,
                                                                                                 test_start_vals,
                                                                                                 test_cat_vals)

    # Get the Batch Tranform Input
    file_ground_truth = 'oos_gt_data.json'
    processed_groundtruth_path = write_to_bucket(file_ground_truth, s3_bucket, s3_prefix, prediction_series,
                                                 prediction_start_values, prediction_cat_values,
                                                 s3_folder='GroundTruth')
    logger.info(f"Ground Truth Input Location: {processed_groundtruth_path}")

    # Downloading ground truth file to container local path
    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/TransformInput").mkdir(parents=True, exist_ok=True)
    
    bucket = s3_bucket
    key = processed_groundtruth_path.split(s3_bucket + '/')[1]
    fn = f"{base_dir}/TransformInput/oos_gt_data.json"
    s3 = boto3.resource("s3")
    s3.Bucket(s3_bucket).download_file(key, fn)
