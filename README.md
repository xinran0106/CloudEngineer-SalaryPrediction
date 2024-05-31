# Salary Prediction for AI-related Jobs:
**End-to-End Machine Learning Solution on AWS**

### Project Overview
This project aims to analyze trends and determinants of salaries for AI-related jobs from 2021 to 2023. The end-to-end machine learning solution leverages various AWS services for data collection, preprocessing, model building, deployment, and visualization.

*Team Members*:
Wen-Chi Lee
Jeansue Wu
Tzuliang Huang
Xinran Wang


## How to Run
1. **Clone the Repository**:
    ```bash
    git clone https://github.com/leon80533/cloud-final
    ```
    
2. **Navigate to the Project Directory**:
    ```bash
    cd cloud-final
    ```

3. **Build Docker Image**:
    ```bash
    docker build -t your-image-name .
    ```

4. **Run Docker Container**:
    ```bash
    docker run -p 8501:8501 your-image-name
    ```

5. **Access Streamlit Interface**: Open `http://localhost:8501` in your web browser.


<br />

### Objective
To determine the key determinants affecting the salaries of AI-related jobs over the last few years.

### Business Question
What are the trends and determinants of salary for AI-related jobs from 2021 to 2023?

### Key Components
- AWS S3: Storage for raw and processed data.
- AWS ECS: Container orchestration service for deploying Docker applications.
- AWS EC2: Compute service for running the models.
- AWS Quicksight: Business intelligence tool for data visualization.

### Architecture Diagram
![alt text](https://github.com/leon80533/cloud-final/blob/main/diagram.png)

### Cost Estimator
1. AWS S3 Bucket
    * Raw: S3 Standard Storage (1 GB) per month - 0.04 USD
    * Results: S3 Standard Storage (1 GB) per month - 0.04 USD
2. AWS ECR
    * Data stored (800 MB per month) - 0.08 USD
3. AWS Fargate
    * Linux OS, ARM CPU, 1-minute duration, 1 task per day, 20 GB ephemeral storage - 0.00 USD
4. Amazon QuickSight
    * 22 working days per month, 10 GB SPICE capacity, 1 author, 5 readers - 31.40 USD

#### Total Monthly Cost Estimate: 31.52 USD

### Data Sources and Pre-processing
*Mapping:*
Mapped job titles into broad categories (e.g., Data Scientist, Data Analyst, Machine Learning Engineer).

*Outliers:*
Removed specialized positions and major outliers.

*Filtering:*
Curated dataset for full-time roles.

### Data Storage
Raw data stored in S3 Raw data bucket. Results stored in S3 Results data bucket. 

### QuickSight EDA
1. Company Size and Salary: Medium-sized companies offer the highest average salaries.
2. Job Title and Salary: Analytical Sales and Research Scientist roles have the highest average salaries.
3. Experience Level Distribution: The majority of professionals are Senior level (SE), followed by Mid-level (MI), Entry-level (EN), and Executive (EX).
4. Salary Distribution: Most salaries cluster between $100,000 and $150,000, with a right-skewed distribution.
5. Experience Level and Salary: Executives (EX) earn the highest average salaries.

### Pipeline
Automate the deployment process with a well-defined pipeline to ensure consistent and efficient deployment on AWS.
![alt text](https://github.com/leon80533/cloud-final/blob/main/pipleline.png)

#### Models Used
* Random Forest
* Gradient Boost
* XGBoost

#### Model Performance
| Algorithm       | R² (%) | RMSE      |
|-----------------|--------|-----------|
| Random Forest   | 42.02  | 46865.95  |
| Gradient Boost  | 43.6   | 46221.6   |
| XGBoost         | 42.81  | 46544.86  |

#### Deployment Steps
1. Build Docker Container: Build and configure the application within a Docker container.
2. Upload to ECR: Push the Docker image to Amazon Elastic Container Registry (ECR).
3. Create ECS Cluster: Set up an ECS cluster to manage and orchestrate the deployment.
4. Create Task Definition: Define resources and configurations for running Docker containers.
5. Service Deployment: Deploy the application services on the ECS cluster.

### Result Artifacts
Generate and store artifacts such as logs and performance metrics to analyze the deployment results and ensure continuous improvement.

### Streamlit Interface
Launch a Streamlit interface for user interaction, providing a user-friendly front end for the deployed application.
![alt text](https://github.com/leon80533/cloud-final/blob/main/streamlit.jpg)

Author:

Xinran Wang,

Wen-Chi Lee, 

Jeansue Wu, 

Tzuliang Huang

# Thank You!
