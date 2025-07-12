# Use a lightweight official Python image
FROM python:3.10-slim-bullseye

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/uploads /app/data

# Expose the port the app runs on
EXPOSE 5000

# Set the command to run the web app
CMD ["python3", "web_app.py"]
