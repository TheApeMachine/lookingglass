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
    build-essential \
    cmake \
    pkg-config \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-all-dev \
    libblas-dev \
    libatlas-base-dev \
    gfortran \
    git \
    ffmpeg \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install a newer version of CMake (3.25+) via direct download
RUN wget https://github.com/Kitware/CMake/releases/download/v3.25.0/cmake-3.25.0-linux-x86_64.tar.gz && \
    tar -xzf cmake-3.25.0-linux-x86_64.tar.gz && \
    mv cmake-3.25.0-linux-x86_64 /opt/cmake && \
    cp -r /opt/cmake/bin/* /usr/local/bin/ && \
    cp -r /opt/cmake/share/* /usr/local/share/ && \
    rm cmake-3.25.0-linux-x86_64.tar.gz

# Update PATH to prioritize new CMake
ENV PATH="/usr/local/bin:/opt/cmake/bin:$PATH"

# Verify CMake installation and ensure it's the right version
RUN cmake --version && which cmake

# Create and activate a virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:/opt/cmake/bin:$PATH"

# Set environment variables for dlib compilation
ENV DLIB_USE_CUDA=1
ENV CMAKE_BUILD_TYPE=Release
ENV CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5"

# Upgrade pip and install wheel first for better package handling
RUN pip install --upgrade pip setuptools wheel

# Copy requirements and install in chunks to avoid timeout
COPY requirements.txt /app/requirements.txt
WORKDIR /app

# Install core packages first with exact versions to avoid conflicts
RUN pip install --timeout=600 --retries=3 \
    flask gunicorn werkzeug wheel setuptools \
    numpy==1.24.3 pillow opencv-python-headless

# Install PyTorch packages (large download) - these are compatible with numpy 1.24.3
RUN pip install --timeout=1200 --retries=5 --no-cache-dir \
    torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118

# Install TensorFlow packages (compatible versions)
RUN pip install --timeout=1200 --retries=5 --no-cache-dir \
    tensorflow==2.13.0 keras==2.13.1

# Debug: Check CMake version before dlib installation
RUN cmake --version && which cmake

# Install face_recognition and related packages
RUN pip install --timeout=1200 --retries=3 --no-cache-dir \
    face_recognition==1.3.0

# Install remaining ML packages with compatible versions
RUN pip install --timeout=600 --retries=3 \
    retina-face==0.0.17 faiss-gpu==1.7.2

# Install remaining utility packages
RUN pip install --timeout=600 --retries=3 \
    qdrant-client moviepy imageio imageio-ffmpeg speechbrain \
    watchdog minio rq redis

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
    libatlas3-base \
    libgfortran5 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from the builder stage.
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY . /app/
WORKDIR /app

# Add the venv to the PATH and set python unbuffered
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Set environment variables to help with GPU detection
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Create directories for data persistence
RUN mkdir -p /app/known_faces /app/crawled_faces

# Make entrypoint script executable
RUN chmod +x /app/entrypoint.sh

EXPOSE 5001

# Use entrypoint script and gunicorn to run the app
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "main:app"]
