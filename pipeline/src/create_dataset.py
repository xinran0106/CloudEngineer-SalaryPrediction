"""
This module provides functionality to create and save datasets from raw data files
using pandas and numpy libraries. It includes functions to read data from a specified 
file path, process the data into structured pandas DataFrames, 
and save these DataFrames to CSV files. Logging is used to handle and report errors.
"""

import logging
import pandas as pd

logger = logging.getLogger("clouds")

def create_dataset(file_path, columns):
    """
    Create a structured dataset from the raw data file.

    Parameters:
    - file_path (str): Path to the raw data file.
    - columns (list): List of column names to include in the DataFrame.

    Returns:
    - DataFrame: A structured DataFrame containing the processed data.
    """
    try:
        data = pd.read_csv(file_path, usecols=columns)
        logger.info('Dataset created successfully from %s', file_path)
        return data
    except Exception as e:
        logger.error('Failed to create dataset from %s: %s', file_path, e)
        raise

def save_dataset(data, output_path):
    """
    Save the dataset to a CSV file.

    Parameters:
    - data (DataFrame): The DataFrame to save.
    - output_path (str): The file path to save the dataset to.
    """
    try:
        data.to_csv(output_path, index=False)
        logger.info('Dataset saved successfully to %s', output_path)
    except Exception as e:
        logger.error('Failed to save dataset to %s: %s', output_path, e)
        raise
