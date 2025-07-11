# Use a standard Python image
FROM bitnami/minideb:bookworm

# Install system dependencies
RUN install_packages wget gnupg nodejs python3-full python3-pip

# Set the working directory
WORKDIR /app

RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Copy and install Python dependencies
COPY requirements.txt .
RUN . /venv/bin/activate \
    && pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

# Install Playwright browsers (with Patchright stealth capabilities)
RUN . /venv/bin/activate && patchright install
RUN . /venv/bin/activate && patchright install-deps

# Copy the entire Scrapy project into the container
COPY . .

# Make entrypoint executable
RUN chmod +x /app/entrypoint.sh

ENV PYTHONUNBUFFERED=1

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]