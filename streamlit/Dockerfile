FROM --platform=linux/arm64/v8 python:3.11 AS build

# Set the working directory inside the container
WORKDIR /app

# Update pip to use the latest version
RUN /usr/local/bin/python -m pip install --upgrade pip

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install the required dependencies
RUN pip install -r requirements.txt

# Copy the Streamlit application files to the working directory
COPY . .

# Expose port 80 for http traffic
EXPOSE 80

# Required to get all streamlit logs in docker
ENV PYTHONUNBUFFERED=1

# Set the command to run the Streamlit application
CMD ["streamlit", "run", "--server.port=80", "--server.fileWatcherType=none", "streamlit_salary.py"] 
