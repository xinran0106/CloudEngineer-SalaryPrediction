import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import logging.config
import os
import boto3
from pathlib import Path
import yaml

# Logging configuration
logging.config.fileConfig("config/logging/local.conf")
logger = logging.getLogger("streamlit")

st.set_page_config(page_title="Salary Prediction App", page_icon="💼", layout="wide")

# Load configuration from YAML file
config_file = "config/config.yaml"
with open(config_file, 'r') as file:
    config = yaml.safe_load(file)

MODEL_BUCKET_NAME = config['s3_buckets']['model_bucket']
DATA_BUCKET_NAME = config['s3_buckets']['data_bucket']
CONFIG_BUCKET_NAME = config['s3_buckets']['config_bucket']
MODEL_FILES = config['models']['files']
DATA_FILE = config['data']['file']
MODEL_PATH = config['models']['path']

# Create artifacts directory to keep model files and data files downloaded from S3 bucket
artifacts = Path() / "artifacts"
artifacts.mkdir(exist_ok=True)

# Download files from S3
def download_s3(bucket_name: str, object_key: str, local_file_path: Path):
    s3 = boto3.client("s3")
    logger.info("Fetching Key: %s from S3 Bucket: %s", object_key, bucket_name)
    try:
        s3.download_file(bucket_name, object_key, str(local_file_path))
        logger.debug("File downloaded successfully to %s", local_file_path)
    except FileNotFoundError as err:
        logger.error("Error downloading file: %s", err)

@st.cache_data
def load_config(s3_key, config_file):
    download_s3(CONFIG_BUCKET_NAME, s3_key, config_file)
    with config_file.open() as file:
        return yaml.load(file, Loader=yaml.SafeLoader)

@st.cache_data
def load_data(s3_key, data_file):
    download_s3(DATA_BUCKET_NAME, s3_key, data_file)
    df = pd.read_csv(data_file)
    return df

@st.cache_resource
def load_model(s3_key, model_file):
    download_s3(MODEL_BUCKET_NAME, s3_key, model_file)
    loaded_model = joblib.load(model_file)
    return loaded_model

def main():
    st.title("💼 Salary Prediction App")
    st.write("""
    ### Predict your salary based on your experience, education, and other factors.
    Use this app to see how much you might earn in different scenarios. Select a model and input your details to get started.
    """)

    s3_models_path = MODEL_PATH

    model_filename1 = 'XGBRegressor_model.joblib'
    model_filename2 = 'GradientBoostingRegressor_model.joblib'
    model_filename3 = 'DecisionTreeRegressor_model.joblib'

    model_s3key1 = s3_models_path + "/" + model_filename1
    model_s3key2 = s3_models_path + "/" + model_filename2
    model_s3key3 = s3_models_path + "/" + model_filename3

    xgb_model = load_model(model_s3key1, artifacts / model_filename1)
    gbr_model = load_model(model_s3key2, artifacts / model_filename2)
    dtr_model = load_model(model_s3key3, artifacts / model_filename3)

    data = load_data(DATA_FILE, artifacts / Path(DATA_FILE).name)
    data = data.drop(['salary_in_usd'], axis=1)
    
    label_encoders = {}
    original_categorical_values = {}
    for column in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        le.fit(data[column])
        label_encoders[column] = le
        original_categorical_values[column] = le.classes_

    def predict_salary(model, features):
        return model.predict([features])[0]

    col1, col2 = st.columns(2)

    with col1:
        st.header("Input Features")
        features = []
        decoded_features = {}
        for col in data.columns:
            if col == 'salary_in_usd':
                continue
            if data[col].dtype == 'object':
                value = st.selectbox(f'{col}', original_categorical_values[col])
                encoded_value = label_encoders[col].transform([value])[0]
                features.append(encoded_value)
                decoded_features[col] = value
            else:
                value = st.number_input(f'{col}', value=float(data[col].mean()))
                features.append(value)
                decoded_features[col] = value

    with col2:
        st.header("Model Selection")
        model_option = st.selectbox('Choose a model', ('XGBRegressor', 'GradientBoostingRegressor', 'DecisionTreeRegressor'))

        if model_option == 'XGBRegressor':
            selected_model = xgb_model
        elif model_option == 'GradientBoostingRegressor':
            selected_model = gbr_model
        else:
            selected_model = dtr_model

        if st.button('Predict Salary'):
            salary = predict_salary(selected_model, features)
            st.write(f'Predicted Salary: ${salary:.2f}')
            
if __name__ == "__main__":
    main()
