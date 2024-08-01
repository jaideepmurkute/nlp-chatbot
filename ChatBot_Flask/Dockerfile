# Use an official Python runtime as a parent image
FROM python:3.12.4-slim AS base

# Define environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=development

# Set the working directory in the container
WORKDIR /app

# Install any needed packages
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Make port 5000 available to the world outside this container
EXPOSE 5100

# Run flask command when the container launches
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=5100"]