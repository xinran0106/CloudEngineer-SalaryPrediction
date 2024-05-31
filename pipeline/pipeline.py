"""
This module orchestrates the pipeline for acquiring, cleaning, and processing cloud data, 
including feature generation, model training, and evaluation, with results saved and uploaded 
to AWS S3.
"""

import argparse
import datetime
import logging.config
from pathlib import Path

import yaml
import csv


import src.acquire_data as ad
import src.create_dataset as cd
import src.generate_features as gf
import src.train_model as tm
import src.aws_utils as aws
import src.analysis as eda

logging.config.fileConfig("config/logging/local.conf")
logger = logging.getLogger("clouds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Acquire, clean, and create features from clouds data"
    )
    parser.add_argument(
        "--config", default="config/config.yaml", help="Path to configuration file"
    )
    args = parser.parse_args()

    # Load configuration file for parameters and run config
    with open(args.config, "r", encoding="utf-8") as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.error.YAMLError as e:
            logger.error("Error while loading configuration from %s", args.config)
        else:
            logger.info("Configuration file loaded from %s", args.config)

    run_config = config.get("run_config", {})

    # Set up output directory for saving artifacts
    now = int(datetime.datetime.now().timestamp())
    artifacts = Path(run_config.get("output", "artifacts")) / str(now)
    artifacts.mkdir(parents=True)

    # Save config file to artifacts directory for traceability
    with (artifacts / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.dump(config, f)

    # Acquire data from S3 bucket and save to disk
    bucket_name = run_config.get("s3_bucket", "cloud-raw")
    s3_key = run_config.get("s3_key", "Data Science Salary 2021 to 2023.csv")
    local_data_path = artifacts / "clouds.csv"
    ad.download_file_from_s3(bucket_name, s3_key, local_data_path)
    

    # Create structured dataset from raw data; save to disk
    data = cd.create_dataset(local_data_path, config["create_dataset"]['columns'])
    cd.save_dataset(data, artifacts / "clouds.csv")
    
    # Generate statistics and visualizations for summarizing the data; save to disk
    figures = artifacts / "figures"
    figures.mkdir()
    eda.save_figures(data, figures)
    
    # Enrich dataset with features for model training; save to disk
    features, label_encoders, org_df = gf.generate_features(data)
    cd.save_dataset(features, artifacts / "features.csv")
    cd.save_dataset(org_df, artifacts / "org_df.csv")

    # Split data into train/test set and train models based on config; save each to disk
    results, train, test = tm.train_models(features, config["train_model"])
    tm.save_data(train, test, artifacts)

    # Save all models and their results
    for model_name, result in results.items():
        tm.save_model(result['best_model'], artifacts / f"{model_name}_model.joblib")

    # Save results to CSV
    tm.save_results(results, artifacts)
    

    # Upload all artifacts to S3
    aws_config = config.get("aws")
    aws.upload_artifacts(artifacts, aws_config['bucket_name'], "artifacts")
