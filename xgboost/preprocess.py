"""Feature engineers the bcm dataset."""
# from pipelines._utils import read_config
import argparse
import logging
import pathlib
import requests
import tempfile
import configparser
import os 
import ast 
import json
import csv
import boto3
import numpy as np
import pandas as pd
from glob import glob
import warnings
warnings.filterwarnings('ignore')

from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


# Reading Configuration file
config = configparser.ConfigParser()
config.read('/opt/ml/processing/config/configurations.ini')

s3_bucket = config.get('BCM_Properties', 's3_bucket')
s3_prefix = config.get('BCM_Properties', 's3_prefix')
key = config.get('BCM_Properties', 'key')
logger.info(key)

train_folder= config.get('BCM_Properties', 'train_folder')
train_file= config.get('BCM_Properties', 'train_file')

validation_folder= config.get('BCM_Properties', 'validation_folder')
validation_file = config.get('BCM_Properties', 'validation_file')

test_folder = config.get('BCM_Properties', 'test_folder')
test_file = config.get('BCM_Properties', 'test_file')

features = ast.literal_eval(config.get('BCM_Properties', 'features'))
target = config.get('BCM_Properties', 'target')

train_percent = int(config.get('BCM_Properties', 'train_percent'))
val_percent = int(config.get('BCM_Properties', 'val_percent'))
test_percent = int(config.get('BCM_Properties', 'test_percent'))

# Lambda 
# function_arn = 'arn:aws:lambda:eu-west-1:610914939903:function:psycopg2_layer'
# region_name ='eu-west-1'
function_arn = config.get('BCM_Properties', 'function_arn')
region_name = config.get('BCM_Properties', 'region_name')

# INPUT DATA
SRC_TS = glob("/opt/ml/processing/input_data/*.csv")[-1]

def fetch_data_from_lambda_and_save_to_csv(s3_bucket, s3_prefix):
    # Create a Boto3 client for Lambda
    lambda_client = boto3.client('lambda', region_name=region_name)

    # Define the Function ARN
    function_arn = function_arn

    # Invoke the Lambda function
    response = lambda_client.invoke(
        FunctionName=function_arn,
        InvocationType='RequestResponse'
    )

    # Retrieve the response payload
    payload = response['Payload'].read()

    # Process the payload
    data = json.loads(payload)

    # Access the retrieved data
    rows = data['body']['data']
    col_names = data['body']['columns']

    # Export the data to a CSV file
    file_path = 'dcm_db_data_final.csv'
    with open(file_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(col_names)  # Write column names to the CSV file
        writer.writerows(rows)  # Write data rows to the CSV file

    # Upload the CSV file to Amazon S3
    s3 = boto3.client('s3')
    s3.upload_file(file_path, s3_bucket, s3_prefix + '/' + file_path)
    key = s3_prefix+ '/'+ file_path
    logger.info(f"Key for DB data : {key}")
    return key

# Encoding Categorical variable
def encode_categorical(df, column ):
    # Encoding categorical column 'product_title'
    label_encoder = preprocessing.LabelEncoder()
    encoded_col = column + "_code"
    df[encoded_col] = label_encoder.fit_transform(df[column])

    # Create dict with product mapping
    product_mapping = {}
    ids = df[column].unique().tolist()
    df.drop(column, axis=1, inplace=True)
    

    for id in ids:
        product_mapping[id] = int(label_encoder.transform([id])[0])
    logger.info(f"Product Mapping for the original product titles to encoded titles:\n{product_mapping}")

    product_mapping_file = 'product_mapping_' + column + '.json'
    with open(product_mapping_file, 'w') as pmjson:
        pmjson.write(json.dumps(product_mapping))

    #  Uploading file to s3 Bucket   
    s3_folder = 'ProductMapping'
    s3 = boto3.client("s3")
    s3.upload_file(product_mapping_file, s3_bucket, s3_prefix + "/" + s3_folder + "/" + product_mapping_file)
    File_location = "s3://" + s3_bucket + "/" + s3_prefix + "/" + s3_folder + "/" + product_mapping_file
    logger.info(f"File upload to S3 Location:{File_location}")
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
    

    #  Encode categorical Variable   
    logger.info("Encoding Categorical Variables..")
    df = encode_categorical(df, 'asin')
    df = encode_categorical(df, 'ad_group_name')
    
    # df = df.assign(campaign_name=df['campaign_name'].str.split('_').str[1])
    df['targeting_type'].replace(['Automatic targeting', 'Manual targeting', 'context'], [0, 1, 2], inplace=True)
    df['status'].replace(['PAUSED', 'ENABLED'], [0, 1], inplace=True)
    df.dropna(inplace=True)
    df['asin_code'] = df['asin_code'].astype(int)
    df['ad_group_name_code'] = df['ad_group_name_code'].astype(int)
    df['targeting_type'] = df['targeting_type'].astype(int)
    
    #  Rearranging columns 
    df1 = df.copy()
    x_cols = features
    y_cols = [target]
    cols = [*y_cols, *x_cols]
    # cols = feature_cols
    print(type(cols))
    print(cols)
    model_data = df1.copy()
    model_data = model_data[cols]
    
    return model_data
    
def train_test_split(model_data, train_percent, val_percent, train_file, validation_file, test_file):
    train_ratio = train_percent *  0.01
    val_ratio = (train_percent + val_percent) * 0.01
    train_data, validation_data, test_data = np.split(
    model_data.sample(frac=1, random_state=1729),
    [int(train_ratio * len(model_data)), int(val_ratio * len(model_data))],
    ) 
    # Oversampling
    frames = [train_data, train_data]
    df = pd.concat(frames)
    for i in range(3):
        train_data = pd.concat([train_data, train_data])
    logger.info(f"train :{train_data.shape}, validation:{validation_data.shape}, test:{test_data.shape}")
    # write to csv  
    train_data.to_csv(f"{base_dir}/train/train.csv", header=True, index=False)
    validation_data.to_csv(f"{base_dir}/validation/validation.csv", header=True, index=False)
    test_data.to_csv(f"{base_dir}/test/test.csv", header=False, index=False)
    return train_data, validation_data, test_data


def upload_files(s3_bucket, s3_prefix, train_folder, train_file, validation_folder, validation_file, test_folder, test_file):
    # Uploading the train file to S3 bucket
    boto3.Session().resource("s3").Bucket(s3_bucket).Object(
        os.path.join(s3_prefix, f"{train_folder}/{train_file}")
    ).upload_file(f"{base_dir}/train/train.csv")
    # ).upload_file(train_file)
    
    train_key = os.path.join(s3_prefix, f"{train_folder}/{train_file}")
    logger.info(f"Train S3 URI: s3://{s3_bucket}/{train_key}")

    # Uploading the validation file to S3 bucket      
    boto3.Session().resource("s3").Bucket(s3_bucket).Object(
        os.path.join(s3_prefix, f"{validation_folder}/{validation_file}")
    ).upload_file(f"{base_dir}/validation/validation.csv")
    # ).upload_file(validation_file)
    
    val_key = os.path.join(s3_prefix, f"{validation_folder}/{validation_file}")
    logger.info(f"Validation S3 URI: s3://{s3_bucket}/{val_key}")
    
    # Uploading the validation file to S3 bucket      
    boto3.Session().resource("s3").Bucket(s3_bucket).Object(
        os.path.join(s3_prefix, f"{test_folder}/{test_file}")
    ).upload_file(f"{base_dir}/test/test.csv")
    # ).upload_file(test_file)
    
    test_key = os.path.join(s3_prefix, f"{test_folder}/{test_file}")
    logger.info(f"Test S3 URI: s3://{s3_bucket}/{test_key}")
    
    

if __name__ == "__main__":
    logger.debug("Starting preprocessing.")
    # logger.info("Downloading data from bucket: %s, key: %s", s3_bucket, key)
    # input_key = fetch_data_from_lambda_and_save_to_csv(s3_bucket, s3_prefix)
    input_key = key
    raw_df = pd.read_csv(SRC_TS, parse_dates=True)
    
    # s3 = boto3.client('s3')
    # obj = s3.get_object(
    #     Bucket=s3_bucket,
    #     Key=input_key
    # )
    # raw_df = pd.read_csv(obj['Body'], parse_dates=True)
    # print(f"Reading data : {raw_df.head(2)}\n shape:{raw_df.shape}")
    df = raw_df.copy()
    
    logger.info(f"Columns: {df.columns}")
    # Upload data to local path    
    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    # input_data = args.input_data
    bucket = s3_bucket
    key = key
    fn = f"{base_dir}/data/prod_disp_one_to_one.csv"
    s3 = boto3.resource("s3")
    s3.Bucket(s3_bucket).download_file(key, fn)
       
    #  Start Data preprocessing   
    logger.info("Starting feature engineering.")
    model_data = feature_engineering(df, features, target)
    logger.info(f'Feature engineering completed :{model_data.columns.tolist()}')
    
    logger.info("Starting Train Test split")
    train_data, validation_data, test_data = train_test_split(model_data, train_percent, val_percent, train_file, validation_file, test_file)
    
    logger.info("Uploading files to S3 Bucket")
    upload_files(s3_bucket, s3_prefix, train_folder, train_file, validation_folder, validation_file, test_folder, test_file)
    
