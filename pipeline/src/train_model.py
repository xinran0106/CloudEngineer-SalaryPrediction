import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import joblib

logger = logging.getLogger("clouds")

def train_models(features, config):
    """
    Train multiple machine learning models using GridSearchCV.

    Parameters:
    - features (DataFrame): The feature matrix for training.
    - config (dict): Configuration dictionary for training the models.

    Returns:
    - results (dict): A dictionary containing the results for each model.
    - train (DataFrame): Training data.
    - test (DataFrame): Test data.
    """
    try:
        # Extract target variable
        target = features['salary_in_usd']
        features = features.drop(['salary_in_usd'], axis=1)
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(features, target, 
                                                            test_size=config['train_test_split']['test_size'], 
                                                            random_state=config['train_test_split']['random_state'])

        # Define the models and their parameter grids
        models = {
            'GradientBoostingRegressor': GradientBoostingRegressor(random_state=123),
            'DecisionTreeRegressor': DecisionTreeRegressor(random_state=42),
            'XGBRegressor': xgb.XGBRegressor(objective='reg:squarederror')
        }

        # Initialize results dictionary
        results = {}

        # Train each model
        for model_name, model in models.items():
            param_grid = config['parameters'][model_name]
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_score = np.sqrt(-grid_search.best_score_)

            logger.info(f"{model_name} Best Parameters: {best_params}")
            logger.info(f"{model_name} Best CV Score (RMSE): {best_score}")

            predictions = best_model.predict(X_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, predictions))
            test_r2 = r2_score(y_test, predictions)

            logger.info(f"{model_name} Test RMSE: {test_rmse}")
            logger.info(f"{model_name} Test R2 Score: {test_r2}")

            results[model_name] = {
                'best_model': best_model,
                'best_params': best_params,
                'best_score': best_score,
                'test_rmse': test_rmse,
                'test_r2': test_r2
            }

        # Combine train and test data with their respective labels for saving
        train = pd.concat([X_train, y_train], axis=1)
        test = pd.concat([X_test, y_test], axis=1)

        return results, train, test
    except Exception as e:
        logger.error('Failed to train models: %s', e)
        raise

def save_model(model, output_path):
    """
    Save the trained model to a file.

    Parameters:
    - model (estimator): The trained model to save.
    - output_path (str): The file path to save the model to.
    """
    try:
        joblib.dump(model, output_path)
        logger.info('Model saved successfully to %s', output_path)
    except Exception as e:
        logger.error('Failed to save model to %s: %s', output_path, e)
        raise

def save_data(train, test, artifacts):
    """
    Save the train and test datasets to CSV files.

    Parameters:
    - train (DataFrame): Training data.
    - test (DataFrame): Test data.
    - artifacts (Path): The directory to save the datasets to.
    """
    try:
        train.to_csv(artifacts / "train_data.csv", index=False)
        test.to_csv(artifacts / "test_data.csv", index=False)
        logger.info('Train and test datasets saved successfully.')
    except Exception as e:
        logger.error('Failed to save datasets: %s', e)
        raise

def save_results(results, artifacts):
    """
    Save the model results to CSV files.

    Parameters:
    - results (dict): The dictionary containing the results for each model.
    - artifacts (Path): The directory to save the results to.
    """
    try:
        for model_name, result in results.items():
            results_df = pd.DataFrame([{
                'best_params': result['best_params'],
                'best_score': result['best_score'],
                'test_rmse': result['test_rmse'],
                'test_r2': result['test_r2']
            }])
            results_df.to_csv(artifacts / f"{model_name}_results.csv", index=False)
            logger.info(f'{model_name} results saved successfully to {artifacts / f"{model_name}_results.csv"}')
    except Exception as e:
        logger.error('Failed to save results: %s', e)
        raise
