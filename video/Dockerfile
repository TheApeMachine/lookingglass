# ---- Builder Stage ----
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS builder

# Install build dependencies including python3-venv
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    cmake \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    imagemagick \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt

# ---- Final Stage ----
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    libopenblas0 \
    liblapack3 \
    libx11-6 \
    libgtk-3-0 \
    imagemagick \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages

COPY . /app/
WORKDIR /app

CMD ["python3", "main.py"]
