# ---- Builder Stage ----
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS builder

# Install build dependencies including python3-venv
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    ffmpeg \
    cmake \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python packages
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip3 install --no-cache-dir -r requirements.txt

# ---- Final Stage ----
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    ffmpeg \
    libopenblas0 \
    liblapack3 \
    libx11-6 \
    libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY . /app/
WORKDIR /app

# Set PATH to use the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

CMD ["python3", "main.py"]
