import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

logger = logging.getLogger("clouds")

def save_figures(df, figures_path):
    """
    Generate and save various EDA figures from the dataset.

    Parameters:
    - df (DataFrame): The input DataFrame with features.
    - figures_path (Path): The directory to save the figures to.
    """
    try:
        # Set seaborn style for plots
        sns.set(style="whitegrid")

        # Group by year and job title, and count occurrences
        job_title_counts_by_year = df.groupby(['work_year', 'job_title']).size().unstack(fill_value=0)

        # Plotting Job Title Counts by Year
        plt.figure(figsize=(20, 8))
        colors = ['blue', 'green', 'red', 'purple', 'orange']  # Define a list of colors for each year

        for i, year in enumerate(job_title_counts_by_year.index):
            plt.plot(job_title_counts_by_year.columns, job_title_counts_by_year.loc[year], label=str(year), color=colors[i % len(colors)])

        plt.title('Job Title Counts by Year')
        plt.xlabel('Job Title')
        plt.ylabel('Count')
        plt.legend(title='Year')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(figures_path / 'job_title_counts_by_year.png')
        plt.close()

        # Data Cleaning: Removing any negative salaries as they are likely errors or outliers
        df_cleaned = df[df['salary_in_usd'] > 0]

        # Rechecking descriptive statistics after cleaning
        salary_stats_cleaned = df_cleaned['salary_in_usd'].describe()

        # Log cleaned salary statistics
        logger.info(f"Cleaned Salary Statistics: \n{salary_stats_cleaned}")

        # Distribution of salary_in_usd
        plt.figure(figsize=(12, 6))
        sns.histplot(df_cleaned['salary_in_usd'], bins=30, kde=True)
        plt.title('Distribution of Salaries in USD')
        plt.xlabel('Salary in USD')
        plt.ylabel('Frequency')
        plt.savefig(figures_path / 'salary_distribution.png')
        plt.close()

        # Box plot for salary_in_usd by experience_level for each work_year
        plt.figure(figsize=(14, 7))
        sns.boxplot(x='work_year', y='salary_in_usd', hue='experience_level', data=df_cleaned)
        plt.title('Salary Distribution by Experience Level Across Work Years')
        plt.xlabel('Work Year')
        plt.ylabel('Salary in USD')
        plt.legend(title='Experience Level')
        plt.savefig(figures_path / 'salary_distribution_by_experience.png')
        plt.close()

        # Box plot for salary_in_usd by company size
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='company_size', y='salary_in_usd', data=df, order=['S', 'M', 'L'])
        plt.title('Salary in USD by Company Size')
        plt.xlabel('Company Size')
        plt.ylabel('Salary in USD')
        plt.savefig(figures_path / 'salary_by_company_size.png')
        plt.close()

        # Box plot for salary_in_usd by company location
        plt.figure(figsize=(18, 8))
        sns.boxplot(x='company_location', y='salary_in_usd', data=df)
        plt.title('Salary in USD by Company Location')
        plt.xticks(rotation=90)
        plt.xlabel('Company Location')
        plt.ylabel('Salary in USD')
        plt.savefig(figures_path / 'salary_by_company_location.png')
        plt.close()

        logger.info('Figures saved successfully.')
    except Exception as e:
        logger.error('Failed to generate and save figures: %s', e)
        raise
