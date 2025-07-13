# ---- Builder Stage ----
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    python3-dev \
    build-essential \
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

# Install newer CMake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.25.0/cmake-3.25.0-linux-x86_64.tar.gz \
    && tar -xzf cmake-3.25.0-linux-x86_64.tar.gz \
    && mv cmake-3.25.0-linux-x86_64 /opt/cmake \
    && ln -sf /opt/cmake/bin/cmake /usr/local/bin/cmake \
    && rm cmake-3.25.0-linux-x86_64.tar.gz

ENV DLIB_USE_CUDA=1
ENV CMAKE_BUILD_TYPE=Release

WORKDIR /app

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

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

ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY . .

EXPOSE 5001

CMD ["gunicorn", "--bind", "0.0.0.0:5001", "main:app"]
