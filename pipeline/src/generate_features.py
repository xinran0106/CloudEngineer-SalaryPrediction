import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger("clouds")

def categorize_job_title(job_title):
    """
    Categorize the job title into predefined categories.
    """
    data_scientist_keywords = ['Data Scientist', 'Data Science Manager', 'Data Science Lead',
                               'Director of Data Science', 'Data Science Consultant',
                               'Data Strategist', 'Data Modeler', 'Head of Data Science']
    data_analyst_keywords = ['Compliance Data Analyst', 'Data Analyst', 'Data Quality Analyst',
                             'Data Analytics Manager', 'Data Specialist', 'Insight Analyst',
                             'Data Operations Analyst', 'Data Analytics Lead', 'Data Analytics Specialist',
                             'Data Analytics Consultant']
    ml_engineer_keywords = ['Applied Machine Learning Engineer', 'Machine Learning Engineer', 'ML Engineer',
                            'Machine Learning Software Engineer', 'Machine Learning Developer', 'Machine Learning Research Engineer',
                            'Lead Machine Learning Engineer', 'Machine Learning Infrastructure Engineer', 'Machine Learning Manager',
                            'Principal Machine Learning Engineer', 'NLP Engineer', 'Computer Vision Engineer', 'MLOps Engineer',
                            'Computer Vision Software Engineer', 'Head of Machine Learning']
    applied_scientist_keywords = ['Applied Scientist', 'Applied Machine Learning Scientist',
                                  'Machine Learning Scientist']
    research_scientist_keywords = ['Machine Learning Researcher', 'Research Scientist', 'Deep Learning Researcher', '3D Computer Vision Researcher']
    data_engineer_keywords = ['Data Architect', 'Data Engineer', 'Big Data Engineer', 'Software Data Engineer', 'ETL Engineer',
                              'Data Infrastructure Engineer', 'Data DevOps Engineer', 'Cloud Database Engineer',
                              'Data Operations Engineer', 'Cloud Data Engineer', 'Research Engineer',
                              'Deep Learning Engineer', 'Data Science Engineer', 'ETL Developer', 'Analytics Engineer']
    bi_keywords = ['BI Analyst', 'BI Data Engineer', 'BI Developer', 'Business Intelligence Engineer']
    ai_keywords = ['AI Developer', 'AI Scientist', 'AI Programmer']

    # Convert job title to lowercase for case-insensitive matching
    job_title_lower = job_title.lower()

    if any(keyword.lower() in job_title_lower for keyword in data_scientist_keywords):
        return 'Data Scientist'
    elif any(keyword.lower() in job_title_lower for keyword in data_analyst_keywords):
        return 'Data Analyst'
    elif any(keyword.lower() in job_title_lower for keyword in ml_engineer_keywords):
        return 'Machine Learning Engineer'
    elif any(keyword.lower() in job_title_lower for keyword in applied_scientist_keywords):
        return 'Applied Scientist'
    elif any(keyword.lower() in job_title_lower for keyword in research_scientist_keywords):
        return 'Research Scientist'
    elif any(keyword.lower() in job_title_lower for keyword in data_engineer_keywords):
        return 'Data Engineer'
    elif any(keyword.lower() in job_title_lower for keyword in bi_keywords):
        return 'BI Related'
    elif any(keyword.lower() in job_title_lower for keyword in ai_keywords):
        return 'AI Related'
    else:
        return 'Other'

def generate_features(df):
    """
    Generate features for model training from the dataset.

    Parameters:
    - df (DataFrame): The input DataFrame with raw data.
    - config (dict): Configuration dictionary for feature generation.

    Returns:
    - DataFrame: A DataFrame with generated features.
    """
    try:
        # Filter to only include full-time employment and remove specified job titles
        df = df[df['employment_type'] == 'FT'].drop(columns=['employment_type']).reset_index(drop=True)
        df = df[df['job_title'] != 'Autonomous Vehicle Technician']
        df = df[df['job_title'] != 'Data Science Tech Lead']

        # Apply the categorization function to the 'job_title' column
        df['job_title'] = df['job_title'].apply(categorize_job_title)

        # Drop the salary-related columns as they are not features
        features = df.drop(['salary', 'salary_currency', 'salary_in_usd'], axis=1)
        
        org_df = features.copy()
        org_df['salary_in_usd'] = df['salary_in_usd']
        
        # Initialize a dictionary to keep track of label encoders for each categorical column
        label_encoders = {}
        # Loop over each object type column and apply Label Encoding
        for column in features.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            features[column] = le.fit_transform(features[column])
            label_encoders[column] = le  # Store the label encoder for future inverse transformation if needed

        # Add the target variable back to the features DataFrame for consistency
        features['salary_in_usd'] = df['salary_in_usd']
        

        logger.info('Features generated successfully.')
        return features, label_encoders, org_df
    except Exception as e:
        logger.error('Failed to generate features: %s', e)
        raise
    