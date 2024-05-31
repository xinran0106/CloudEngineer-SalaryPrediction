
# Base image
FROM --platform=linux/arm64/v8 python:3.11 AS build

# Set the working directory in the container
WORKDIR /pipeline

# Copy the current directory contents into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY artifacts/ ./artifacts/
COPY config/ /pipeline/config
COPY pipeline.py /pipeline

# Run pipeline.py when the container launches
CMD ["python", "pipeline.py"]