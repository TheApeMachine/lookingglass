# ---- Builder Stage ----
# Use the official NVIDIA CUDA runtime image as the base.
# It contains the necessary libraries for GPU operations.
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS builder

# Prevent tzdata/etc from prompting for input
ENV DEBIAN_FRONTEND=noninteractive

# Install build-time dependencies: Python, venv for creating an isolated environment,
# and build-essential/cmake for compiling Python packages like dlib.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    python3-dev \
    cmake \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create and activate a virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install Python dependencies into the virtual environment
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir git+https://github.com/ageitgey/face_recognition_models

# ---- Final Stage ----
# Start from the same clean base image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install only the necessary RUNTIME dependencies.
# We no longer need build-essential, cmake, python3-dev, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    libopenblas0 \
    liblapack3 \
    libx11-6 \
    libgtk-3-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from the builder stage.
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY . /app/
WORKDIR /app

# Add the venv to the PATH and set python unbuffered
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Create directories for data persistence
RUN mkdir -p /app/known_faces

CMD ["python3", "embedder_service.py"]
