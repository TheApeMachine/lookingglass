FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy both worker scripts
COPY analyze.py page_worker.py ./

CMD ["python", "analyze.py"] 