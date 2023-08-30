""" Retrieving ground Truth data from the database, processing it and uploading to S3 bucket as Batch Transform Input """
import logging
import pathlib
import os 
import ast 
import json
import configparser

import boto3
import pandas as pd

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


# Initializing Config Parser
config = configparser.ConfigParser()
config.read('/opt/ml/processing/config/configurations.ini')

# Reading the configuration variables
s3_bucket = config.get('BCM_Properties', 's3_bucket')
s3_prefix = config.get('BCM_Properties', 's3_prefix')
features = ast.literal_eval(config.get('BCM_Properties', 'features'))
target = config.get('BCM_Properties', 'target')
ground_truth_folder = config.get('BCM_DriftCheck_Properties', 'ground_truth_folder')
ground_truth_file = config.get('BCM_DriftCheck_Properties', 'ground_truth_file')


ad_group_name_key = config.get('BCM_DriftCheck_Properties', 'ad_group_name_key')
asin_key = config.get('BCM_DriftCheck_Properties', 'asin_key')
# Lambda variables
function_arn = config.get('BCM_Properties', 'function_arn')
lambda_region = config.get('BCM_Properties', 'region_name')

function_arn = 'arn:aws:lambda:eu-west-1:361625399431:function:Pipeline-DCM-aiml-stg'

def fetch_data_from_lambda_and_save_to_csv(function_arn):
    # Create a Boto3 client for Lambda
    lambda_client = boto3.client('lambda', region_name=lambda_region)

  
    # Invoke the Lambda function
    response = lambda_client.invoke(
        # FunctionName="arn:aws:lambda:eu-west-1:361625399431:function:Pipeline-DCM-aiml-stg",
        FunctionName = function_arn,
        InvocationType='RequestResponse'
    )

    # Retrieve the response payload
    payload = response['Payload'].read()  # Retrieve the response payload
    data = json.loads(payload)

    key = data['body']['path']  # Access the retrieved data
    logger.info(f"Ground Truth Key : {key}")
    return key

def read_json_file_from_s3(bucket, key):
    s3_client = boto3.client("s3")
    response = s3_client.get_object(Bucket=bucket, Key=key)
    data = json.load(response['Body'])
    return data

def encode_categorical(df, column_name, encoding_map):
    df[column_name + '_code'] = df[column_name].map(encoding_map)
    return df

def feature_engineering(df, features, target):    
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df['year'] = df.date.dt.year
    df['weekday'] = df.date.dt.dayofweek
    df['day'] = df.date.dt.day
    df['month'] = df.date.dt.month

    # Dropping the date column
    df.drop('date', axis=1, inplace=True)
    
    # Modifying asin and Campaign_name to integer values
    if 'advertised_asin' in df.columns:
        df.rename(columns = {'advertised_asin' : 'asin'}, inplace=True)
    if 'closest_cluster' in df.columns:
        df.rename(columns = {'closest_cluster' : 'Cluster'}, inplace=True)
    #  Encode categorical Variable   
    logger.info("Encoding Categorical Variables..")
    
    # Read the encoding maps from S3 bucket
    ad_group_name_encoding_map = read_json_file_from_s3(s3_bucket, ad_group_name_key)
    asin_encoding_map = read_json_file_from_s3(s3_bucket, asin_key)
    
    df = encode_categorical(df, 'ad_group_name', ad_group_name_encoding_map)
    df = encode_categorical(df, 'asin', asin_encoding_map)
    
    df['targeting_type'].replace(['Automatic targeting', 'Manual targeting', 'context'], [0, 1, 2], inplace=True)
    df['status'].replace(['PAUSED', 'ENABLED'], [0, 1], inplace=True)
    df.dropna(inplace=True)
    df['asin_code'] = df['asin_code'].astype(int)
    df['ad_group_name_code'] = df['ad_group_name_code'].astype(int)
    df['targeting_type'] = df['targeting_type'].astype(int)
    df['Cluster'] = df['Cluster'].astype(int)
    
    df1 = check_outliers(df)
    #  Rearranging columns 
    
    
    x_cols = features
    y_cols = [target]
    cols = [*y_cols, *x_cols]
    # cols = feature_cols
    print(type(cols))
    print(cols)
    model_data = df1.copy()
    model_data = model_data[cols]
    model_data.to_csv(f"{base_dir}/Test/test.csv", header=True, index=False) 
    model_data.to_csv(f"{base_dir}/TransformInput/gt_data.csv", header=False, index=False) 
    return model_data

def upload_files(s3_bucket, s3_prefix, ground_truth_folder, ground_truth_file):
    # Uploading the train file to S3 bucket
    boto3.Session().resource("s3").Bucket(s3_bucket).Object(
        os.path.join(s3_prefix, f"{ground_truth_folder}/{ground_truth_file}")
    ).upload_file(f"{base_dir}/TransformInput/gt_data.csv")
    
    
    ground_truth_key = os.path.join(s3_prefix, f"{ground_truth_folder}/{ground_truth_file}")
    logger.info(f"Ground Truth S3 URI: s3://{s3_bucket}/{ground_truth_key}")
    return ground_truth_key

def check_outliers(df):
    if df[df['spend'] > 500].shape[0] < (0.2 * df.shape[0]):
        df = df[df['spend'] < 500]
    return df

if __name__ == "__main__":
    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/TransformInput").mkdir(parents=True, exist_ok=True)
    fn = f"{base_dir}/TransformInput/gt_data.csv"
    
    
    #  Test output 
    # opt/ml/processing/Test
    pathlib.Path(f"{base_dir}/Test").mkdir(parents=True, exist_ok=True)
    # fn_test = f"{base_dir}/Test/test.csv"
    
    # Fetch latest ground truth data 
    input_path = fetch_data_from_lambda_and_save_to_csv(function_arn)
    input_key = input_path.split(s3_bucket + "/")[1]
    
    
    s3 = boto3.client('s3')
    obj = s3.get_object(
        Bucket=s3_bucket,
        Key=input_key
    )
    df = pd.read_csv(obj['Body'], parse_dates=True)
    processed_ground_truth = feature_engineering(df, features, target)      
    ground_truth_key = upload_files(s3_bucket, s3_prefix, ground_truth_folder, ground_truth_file) # upload to S3 bucket  
    
    s3 = boto3.resource("s3")
    s3.Bucket(s3_bucket).download_file(ground_truth_key, fn)
    
    gt_df = pd.read_csv(fn)
    logger.info("Ground Truth shape: ", gt_df.shape)
