# Use an official Python runtime as a base image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies if needed (e.g., git for model downloads)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Upgrade pip and install the Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of your application code into the container
COPY . /app

# Define the default command to run your script
CMD ["python", "main.py"]
