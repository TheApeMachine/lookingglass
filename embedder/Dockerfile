FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    cmake \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt /app/requirements.txt
WORKDIR /app

RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app/

# Create directories for data persistence
RUN mkdir -p /app/known_faces

CMD ["python3", "embedder.py"]
