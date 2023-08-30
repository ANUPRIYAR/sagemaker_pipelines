"""Feature engineers the out of stock dataset."""
# import sagemaker
import random
import json
import boto3
import numpy as np
import pandas as pd
import ast

from sklearn import preprocessing


import warnings
warnings.filterwarnings('ignore')
import logging

from sklearn import preprocessing

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

interpolation_technique = 'linear'
freq = 'D'
# key = 'oos_prediction/data/Forecast/Formatted_Data/Syn_Timeseries_2013_2023.csv'
key = 'oos_prediction/data/data.csv'
s3_bucket = 'sagemaker-eu-west-1-610914939903'
date_col = 'date'
asin_col = 'asin'
units_sold_col = 'ordered_units'

bucket = 'sagemaker-eu-west-1-610914939903'
target_key = 'oos_prediction/data/Forecast/target'

def upsampling(df, freq, interpolation_technique):
    grouped = df.groupby(asin_col)
    dfs = []
    for title, group in grouped:
        group.set_index(date_col, inplace=True)      
        group = group.resample(freq).mean()    # Resample the data at daily frequency to fill in missing dates 
        # Interpolate the missing values using linear interpolation
        group[units_sold_col] = group[units_sold_col].interpolate(method=interpolation_technique) 
        group[asin_col] = title # Add the product_title column and reset the index
        group = group.reset_index()   
        dfs.append(group)

    df_upsampled = pd.concat(dfs)
    df_upsampled[date_col] = df_upsampled[date_col].dt.strftime('%Y-%m-%d %H:%M:%S')
    # df_upsampled[date_col] = pd.to_datetime(df_upsampled[date_col])
    return df_upsampled

def feature_engineering(df, freq, interpolation_technique):
    print("Extracting features from raw data..")    
    columns = ["date","ordered_units","asin","has_campaign"]
    
    df = df[columns]
    df[date_col] = pd.to_datetime(df[date_col], utc=True)
    
    print(f"Upsampling data with Frequency: {freq} \n interpolation : {interpolation_technique}")
    df_upsampled = upsampling(df, freq, interpolation_technique)
    # df_upsampled[date_col] = df_upsampled[date_col].dt.strftime('%Y-%m-%d %H:%M:%S')
    print(f"Upsampling complete. Shape: {df_upsampled.shape}")
    
    
     # Removing data for the product title with less than 300 rows
    print("Removing asin datasets with less than 300 rows")
    counts = df_upsampled[asin_col].value_counts()
    df_upsampled = df_upsampled[~df_upsampled[asin_col].isin(counts[counts < 300].index)]
    df_upsampled = df_upsampled[columns]
    #  Encode categorical Variable   
    # print("Encoding Categorical Variables..")
    # df_upsampled = encode_categorical(df_upsampled)
    
    # Drop the column asin_col
    # df_upsampled.drop(asin_col, axis=1, inplace=True)
    return df_upsampled



if __name__ == "__main__":  
    
    target_path = '/opt/ml/processing/target'
    test_path = '/opt/ml/processing/test'
    
    s3 = boto3.client('s3')
    obj = s3.get_object(
        Bucket=s3_bucket,
        Key=key
    )
    raw_df = pd.read_csv(obj['Body'], parse_dates=True)
    df = raw_df.copy()
    logger.info("Feature Engineering started...")
    df_upsampled = feature_engineering(df, freq, interpolation_technique)
    logger.info(f"Feature Enginerring completed.\n Dataset shape: {df_upsampled.shape}")
    logger.info(df_upsampled.head())
    # df = df_upsampled.copy()
    
    logger.info("Creating Train and validation dataframe..")
    train_data = df_upsampled[df_upsampled[date_col] < '2022-12-31']
    val_data = df_upsampled[df_upsampled[date_col] >= '2022-12-31']
    logger.info(f"Shape of train dataset: {train_data.shape}\nShape of Test dataset:{val_data.shape}")
    logger.info(train_data.head())
    df = train_data.copy()
    
    df.rename( columns = {'date':'timestamp', 'ordered_units':'target_value','asin':'item_id' , 'has_campaign':'has_cam'}, inplace = True)
    # yyyy-MM-dd hh:mm:ss
    df['timestamp'] = pd.to_datetime(df.timestamp)
    df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d')
    logger.info(df['timestamp'][0])
    df['has_cam'].fillna('False', inplace=True)
    convert_dict = {
                'target_value': float,
                'item_id' : str,
                'has_cam': str    
                }
    
    df = df.astype(convert_dict)
    logger.info(f"df date format :{df['timestamp'][0]}")
    df.to_csv(f"{target_path}/target.csv", index=False, date_format='%Y-%m-%d')
    region = 'eu-west-1'
    session = boto3.Session(region_name=region) 
    
    s3 = session.client('s3')
    s3.upload_file(Filename=f"{target_path}/target.csv", Bucket = bucket, Key = f"{target_key}/target.csv")
    s3_target_path = f"s3://{bucket}/{target_key}/target.csv"
    logger.info(f"File uploaded to path : {s3_target_path}")
    
    #  Processing Test data 
    df_test = val_data.copy()
    df_test.rename( columns = {'date':'timestamp', 'ordered_units':'target_value','asin':'item_id' , 'has_campaign':'has_cam'}, inplace = True)    
    df_test['timestamp'] = pd.to_datetime(df_test.timestamp)
    df_test['timestamp'] =  df_test['timestamp'].dt.strftime('%Y-%m-%d')
    # logger.info(df_test['timestamp'][0])
    df_test['has_cam'].fillna('False', inplace=True)
    convert_dict = {
                'target_value': float,
                'item_id' : str,
                'has_cam': str    
                }
    
    df_test = df_test.astype(convert_dict)
    df_test = df_test[['target_value','timestamp','item_id','has_cam']]
    # logger.info(f"df test date format :{df_test['timestamp'][0]}")
    
    df_test.to_csv(f"{test_path}/test.csv", index=False, date_format='%Y-%m-%d')
    
    
