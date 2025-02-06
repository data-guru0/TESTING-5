# Use a Python base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -e .

# Expose the port if your app needs one (if it's a web app or API)
EXPOSE 5000

# Set the default command to run your app (change this if it's different)
CMD ["python", "application.py"]
