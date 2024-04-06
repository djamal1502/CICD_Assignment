# Using an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copying the current directory contents into the container at /usr/src/app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Run train.py when the container launches to train and save the model
RUN python train.py

# Command to run test.py when the container is run
CMD ["python", "test.py"]
