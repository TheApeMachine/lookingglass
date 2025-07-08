# ---- Builder Stage ----
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies for Python, Chrome, and ChromeDriver
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
    wget \
    gnupg \
    unzip \
    curl \
    # Add Google Chrome repository for the builder stage
    && wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list \
    # Update after adding new repo, then install chrome
    && apt-get update && apt-get install -y --no-install-recommends google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

# Install ChromeDriver
RUN CHROME_MAJOR_VERSION=$(google-chrome --version | awk '{print $3}' | cut -d'.' -f1) \
    && echo "Detected Chrome major version: $CHROME_MAJOR_VERSION" \
    && CHROMEDRIVER_VERSION=$(curl -s "https://googlechromelabs.github.io/chrome-for-testing/LATEST_RELEASE_${CHROME_MAJOR_VERSION}") \
    && echo "Using ChromeDriver version: $CHROMEDRIVER_VERSION" \
    && curl -Lo /tmp/chromedriver.zip "https://storage.googleapis.com/chrome-for-testing-public/${CHROMEDRIVER_VERSION}/linux64/chromedriver-linux64.zip" \
    && unzip /tmp/chromedriver.zip -d /usr/local/bin/ \
    && rm -rf /tmp/chromedriver.zip \
    && mv /usr/local/bin/chromedriver-linux64/chromedriver /usr/local/bin/chromedriver \
    && rm -rf /usr/local/bin/chromedriver-linux64 \
    && chmod +x /usr/local/bin/chromedriver

# Create and activate a virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python packages into the venv
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf ~/.cache/pip

# ---- Final Stage ----
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install only runtime dependencies including Google Chrome
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    libopenblas0 \
    liblapack3 \
    libx11-6 \
    libgtk-3-0 \
    wget \
    gnupg \
    # Add google chrome source and install
    && wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list \
    && apt-get update && apt-get install -y --no-install-recommends google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

# Copy ChromeDriver and the Python venv from builder stage
COPY --from=builder /usr/local/bin/chromedriver /usr/local/bin/chromedriver
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY . /app/
WORKDIR /app

# Add the venv to the PATH and set python unbuffered
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Create directory for downloaded faces
RUN mkdir -p /app/crawled_faces

CMD ["python3", "crawler.py"]
