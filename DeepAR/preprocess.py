"""Feature engineers the out of stock dataset."""
import json
import boto3
import pandas as pd
import ast
import logging
import configparser
from glob import glob
import warnings
warnings.filterwarnings('ignore')

from sklearn import preprocessing

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# Reading Configuration file
config = configparser.ConfigParser()
config.read('/opt/ml/processing/config/configurations.ini')

s3_bucket = config.get('OutOfStock_Properties', 's3_bucket')
s3_prefix = config.get('OutOfStock_Properties', 's3_prefix')
input_prefix = config.get('OutOfStock_Properties', 'input_prefix')
ads_key = config.get('OutOfStock_Properties', 'raw_ads_data')
file_train = config.get('OutOfStock_Preprocess', 'file_train')
file_test = config.get('OutOfStock_Preprocess', 'file_test')

interpolation_technique = config.get('OutOfStock_Preprocess', 'interpolation_technique')
freq = config.get('OutOfStock_Preprocess', 'freq')
features = config.get('OutOfStock_Preprocess', 'features')
date_col = config.get('OutOfStock_Preprocess', 'date_col')
asin_col = config.get('OutOfStock_Preprocess', 'asin_col')
units_sold_col = config.get('OutOfStock_Preprocess', 'units_sold_col')
asin_cat_col = config.get('OutOfStock_Preprocess', 'asin_cat_col')
file_batch_transform = config.get('OutOfStock_Preprocess', 'file_batch_transform')
base_dir = f's3://{s3_bucket}/{s3_prefix}'
# key = 'oos_prediction/data/vc_forecast_and_inventory_ordered_units.csv'
key = f"{s3_prefix}/{input_prefix}"

# INPUT DATA
SRC_TS = glob("/opt/ml/processing/input_data/*.csv")[-1]

def process_data(df):
    result_df = pd.DataFrame()  # Initialize an empty DataFrame for the results
    # Group the DataFrame by product_name and date
    grouped_df = df.groupby(['asin', 'date'])
    # Iterate over each group
    for (product_name, date), group in grouped_df:
        # print(group)
        distributor_views = group['distributor_view'].unique()
        # print(distributor_views)
        if len(distributor_views) == 1:
            # If there is only one unique distributor_view, add the entry to the result DataFrame
            result_df = result_df.append(group)
        else:
          # If 'manufacturing' is one of the unique distributor_views, filter and add that entry
              manufacturing_entry = group[group['distributor_view'] == 'Manufacturing']
              result_df = result_df.append(manufacturing_entry)
    return result_df


# def upsampling(df, freq, interpolation_technique):
#     grouped = df.groupby(asin_col)
#     dfs = []
#     for title, group in grouped:
#         group.set_index(date_col, inplace=True)
#         group = group.resample(freq).sum()  # Resample the data at daily frequency to fill in missing dates 
#         # Interpolate the missing values using linear interpolation
#         group[units_sold_col] = group[units_sold_col].interpolate(method=interpolation_technique)
#         group[asin_col] = title  # Add the product_title column and reset the index
#         group = group.reset_index()
#         dfs.append(group)

#     df_upsampled = pd.concat(dfs)
#     df_upsampled[date_col] = df_upsampled[date_col].dt.strftime('%Y-%m-%d %H:%M:%S')
#     # df_upsampled[date_col] = pd.to_datetime(df_upsampled[date_col])
#     return df_upsampled

def upsampling(df_retail, df_ads, freq, interpolation_technique):
    # Upsampling retail data  
    grouped_retail = df_retail.groupby(asin_col)
    dfs = []
    for title, group in grouped_retail:
        group.set_index(date_col, inplace=True)
        group = group.resample(freq).sum()  # Resample the data at daily frequency to fill in missing dates 
        # Interpolate the missing values using linear interpolation
        group[units_sold_col] = group[units_sold_col].interpolate(method=interpolation_technique)
        group[asin_col] = title  # Add the product_title column and reset the index
        group = group.reset_index()
        dfs.append(group)

    df_upsampled_retail = pd.concat(dfs)
    df_upsampled_retail[date_col] = df_upsampled_retail[date_col].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    #  Upsampling ads data 
    grouped_ads = df_ads.groupby(asin_col)
    df_ads = []
    for title, group in grouped_ads:
        group.set_index(date_col, inplace=True)
        group = group.resample(freq).sum()  # Resample the data at daily frequency to fill in missing dates 
        # Interpolate the missing values using linear interpolation
        group['14_day_total_units'] = group['14_day_total_units'].interpolate(method=interpolation_technique)
        group[asin_col] = title  # Add the product_title column and reset the index
        group = group.reset_index()
        df_ads.append(group)
    df_upsampled_ads = pd.concat(df_ads)
    df_upsampled_ads[date_col] = df_upsampled_ads[date_col].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    df = df_upsampled_retail.merge(df_upsampled_ads, how='left', on= ['date', 'asin'] )
    df['14_day_total_units'].fillna(0, inplace=True)
    df['ordered_units'] = df['ordered_units'] + df['14_day_total_units']
    df.drop('14_day_total_units', axis=1, inplace=True)
    
    File = 'Retail_sales_with_ads.csv'
    df.to_csv(File, index=False)
    s3 = boto3.client("s3")
    s3_folder = 'Processed_Data'
    s3.upload_file(File, s3_bucket, s3_prefix + "/" + s3_folder + "/" + File)
    return df


def encode_categorical(df_upsampled):
    # Encoding categorical column 'product_title'
    label_encoder = preprocessing.LabelEncoder()
    df_upsampled[asin_cat_col] = label_encoder.fit_transform(df_upsampled[asin_col])

    # Create dict with product mapping
    product_mapping = {}
    asin_ids = df_upsampled[asin_col].unique().tolist()

    for id in asin_ids:
        product_mapping[id] = int(label_encoder.transform([id])[0])
    logger.info(f"Product Mapping for the original product titles to encoded titles:\n{product_mapping}")
    logger.info(f"Printing for testing pipeline execution")

    product_mapping_file = 'product_mapping.json'
    with open(product_mapping_file, 'w') as pmjson:
        pmjson.write(json.dumps(product_mapping))

    #  Uploading file to s3 Bucket   
    s3_folder = 'ProductMapping'
    s3 = boto3.client("s3")
    s3.upload_file(product_mapping_file, s3_bucket, s3_prefix + "/" + s3_folder + "/" + product_mapping_file)
    File_location = "s3://" + s3_bucket + "/" + s3_prefix + "/" + s3_folder + "/" + product_mapping_file
    logger.info(f"File upload to S3 Location:{File_location}")
    return df_upsampled


def feature_engineering(df_retail, df_ads, freq, interpolation_technique):
    logger.info("Extracting features from raw data..")
    
    #  Selecting only rows with distributor_view as 'Manufacturing' and Period as 'Daily'
    if 'distributor_view' in df_retail.columns:
        df_retail = process_data(df_retail)
    if 'period' in df_retail.columns:
        df_retail = df_retail[df_retail['period'] == 'DAILY']
    
    columns = ast.literal_eval(features)
    df_retail = df_retail[columns]
    df_retail[date_col] = pd.to_datetime(df_retail[date_col], utc=True)

    logger.info(f"Upsampling data with Frequency: {freq} \n interpolation : {interpolation_technique}")
    df_upsampled = upsampling(df_retail, df_ads, freq, interpolation_technique)
    # df_upsampled = upsampling(df,df_ads freq, interpolation_technique)
    # df_upsampled[date_col] = df_upsampled[date_col].dt.strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"Upsampling complete. Shape: {df_upsampled.shape}")

    # Removing data for the product title with less than 300 rows
    logger.info("Removing asin datasets with less than 29 rows")
    counts = df_upsampled[asin_col].value_counts()
    df_upsampled = df_upsampled[~df_upsampled[asin_col].isin(counts[counts < 29].index)]

    #  Encode categorical Variable   
    logger.info("Encoding Categorical Variables..")
    df_upsampled = encode_categorical(df_upsampled)

    # Drop the column asin_col
    df_upsampled.drop(asin_col, axis=1, inplace=True)
    return df_upsampled

def train_test_split(df_upsampled):
    # Get all unique asins 
    asins = df_upsampled[asin_cat_col].unique().tolist()
    
    train_ts  = []
    val_ts = []
    for asin in asins:
        asin_ts = df_upsampled[df_upsampled[asin_cat_col] == asin]
        asin_ts.reset_index(inplace=True)
        index = int(asin_ts.shape[0] * 0.85)
        train_ts.append(asin_ts[:index])
        val_ts.append(asin_ts[index:])
        
    train_data = pd.concat(train_ts, ignore_index=True)
    val_data = pd.concat(val_ts, ignore_index=True)
    
    train_data.drop('index', axis=1, inplace=True)
    train_data.reset_index(drop=True, inplace=True)
    val_data.drop('index', axis=1, inplace=True)
    val_data.reset_index(drop=True, inplace=True)
    print(f"train shape: {train_data.shape} , val shape : {val_data.shape}")
    return train_data, val_data


# Creating Training and Test data in json format
def get_timeseries(df):
    product_ids = df[asin_cat_col].unique().tolist()
    # df[date_col] = pd.to_datetime(df[date_col], utc=True)
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
    logger.debug("Starting preprocessing.")
    logger.info("Downloading data from bucket: %s, key: %s", s3_bucket, key)
    # s3 = boto3.client('s3')
    # obj = s3.get_object(
    #     Bucket=s3_bucket,
    #     Key=key
    # )
    # raw_df = pd.read_csv(obj['Body'], parse_dates=True)
    raw_df = pd.read_csv(SRC_TS, parse_dates=True)
    
    logger.info(f"INFO : {raw_df.info()}")
    df_retail = raw_df.copy()  
    
    # Ads Data 
    s3 = boto3.client('s3')
    obj = s3.get_object(
        Bucket=s3_bucket,
        Key=ads_key
        )
    df_ads = pd.read_csv(obj['Body'], parse_dates=True)
    logger.info(f"Columns: {df_ads.columns.tolist()}")
    if 'advertised_asin' in df_ads.columns.tolist():
        logger.info("True")
        df_ads.rename(columns ={'advertised_asin' : 'asin'}, inplace=True)
        
    df_ads = df_ads[['date', 'asin', '14_day_total_units']]
    df_ads[date_col] = pd.to_datetime(df_ads[date_col], utc=True)

    logger.info("Feature Engineering started...")
    # df_upsampled = feature_engineering(df, freq, interpolation_technique)
    df_upsampled = feature_engineering(df_retail, df_ads, freq, interpolation_technique)
    logger.info(f"Feature Enginerring completed.\n Dataset shape: {df_upsampled.shape}")

    # Create Train and Test Dataset
    logger.info("Creating Train and validation dataframe..")
    # train_data = df_upsampled[df_upsampled[date_col] < '2022-12-31']
    # val_data = df_upsampled[df_upsampled[date_col] >= '2022-12-31']
    train_data, val_data = train_test_split(df_upsampled)
    logger.info(f"Shape of train dataset: {train_data.shape}\nShape of Test dataset:{val_data.shape}")

    # Create the timeseries data
    timeseries_train, train_start_vals, train_cat_vals = get_timeseries(train_data)
    timeseries_test, test_start_vals, test_cat_vals = get_timeseries(val_data)

    # Create train and test json files with timeseries data and upload it into S3 bucket
    write_to_bucket(file_train, s3_bucket, s3_prefix, timeseries_train, train_start_vals, train_cat_vals,
                    s3_folder='TRAIN')
    write_to_bucket(file_test, s3_bucket, s3_prefix, timeseries_test, test_start_vals, test_cat_vals, s3_folder='TEST')

    # Get Batch Transform input 
    prediction_series, prediction_start_values, prediction_cat_values = get_batchtransform_input(timeseries_train,
                                                                                                 train_cat_vals,
                                                                                                 test_start_vals,
                                                                                                 test_cat_vals)

    prediction_input_path = write_to_bucket(file_batch_transform, s3_bucket, s3_prefix, prediction_series,
                                            prediction_start_values, prediction_cat_values,
                                            s3_folder='BATCH_TRANSFORM/INPUT')
    logger.info(f"Batch Transform Input Location: {prediction_input_path}")
