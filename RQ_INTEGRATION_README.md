# LookingGlass RQ Integration

This document explains the new Redis Queue (RQ) based ingress pipeline for LookingGlass that uses MinIO bucket notifications to automatically trigger facial recognition processing.

## Architecture Overview

The new architecture consists of:

1. **MinIO** - Object storage with bucket notification capabilities
2. **Redis** - Message broker for RQ job queue
3. **Webhook Service** - Receives MinIO notifications and queues RQ jobs
4. **RQ Workers** - Process facial recognition jobs asynchronously
5. **GPU Worker** - Existing facial recognition processing service

## Flow Diagram

```
[Crawler] → [MinIO Upload] → [Bucket Notification] → [Webhook Service] → [RQ Queue] → [RQ Worker] → [GPU Worker] → [Qdrant]
```

## Key Benefits

- **Asynchronous Processing**: No blocking uploads, jobs are processed in the background
- **Scalability**: Multiple RQ workers can process jobs in parallel
- **Reliability**: Failed jobs can be retried automatically
- **Monitoring**: Full visibility into job status and queue health
- **Decoupling**: Upload and processing are completely separate concerns

## Services

### 1. Redis
- **Purpose**: Message broker for RQ
- **Port**: 6379
- **Data persistence**: Yes (with appendonly mode)

### 2. Webhook Service
- **Purpose**: Receives MinIO bucket notifications and queues RQ jobs
- **Port**: 5002
- **Endpoints**:
  - `POST /webhook/minio` - Main webhook endpoint for MinIO notifications
  - `GET /health` - Health check and queue stats
  - `GET /job/<job_id>` - Get status of specific job
  - `GET /queue/stats` - Get queue statistics
  - `POST /test/queue` - Test endpoint for manual job queueing

### 3. RQ Worker
- **Purpose**: Processes queued facial recognition jobs
- **Replicas**: 2 (configurable)
- **Queue**: `facial_recognition`
- **Jobs**:
  - `process_image_upload` - Process uploaded images
  - `process_video_upload` - Process uploaded videos  
  - `cleanup_processed_item` - Handle deleted items

## Setup Instructions

### 1. Start the Services

```bash
# Start all services including new RQ components
docker-compose up -d

# Check if all services are running
docker-compose ps
```

### 2. Configure MinIO Bucket Notifications

```bash
# Make the setup script executable
chmod +x scripts/setup-minio-notifications.sh

# Run the setup script
./scripts/setup-minio-notifications.sh

# Or run with custom configuration
WEBHOOK_ENDPOINT="http://webhook-service:5002/webhook/minio" \
IMAGE_BUCKET="my-images" \
VIDEO_BUCKET="my-videos" \
./scripts/setup-minio-notifications.sh
```

### 3. Verify the Setup

```bash
# Check webhook service health
curl http://localhost:5002/health

# Check queue statistics
curl http://localhost:5002/queue/stats

# Check service logs
docker-compose logs -f webhook-service rq-worker
```

## Testing the Integration

### 1. Upload Test Files

```bash
# Upload an image (should trigger facial recognition)
mc cp /path/to/image.jpg myminio/images/

# Upload a video (should trigger video processing)
mc cp /path/to/video.mp4 myminio/videos/
```

### 2. Monitor Job Processing

```bash
# Check webhook service logs for incoming notifications
docker-compose logs webhook-service

# Check RQ worker logs for job processing
docker-compose logs rq-worker

# Check queue stats via API
curl http://localhost:5002/queue/stats
```

### 3. Manual Job Testing

```bash
# Queue a test job manually
curl -X POST http://localhost:5002/test/queue \
  -H "Content-Type: application/json" \
  -d '{"bucket": "images", "object": "test.jpg", "event": "s3:ObjectCreated:Put"}'
```

## Monitoring and Debugging

### Queue Statistics

```bash
# Get detailed queue information
curl -s http://localhost:5002/queue/stats | jq .

# Example response:
{
  "status": "success",
  "queue_name": "facial_recognition",
  "stats": {
    "queued_jobs": 5,
    "failed_jobs": 0,
    "finished_jobs": 23,
    "started_jobs": 2,
    "deferred_jobs": 0
  }
}
```

### Job Status

```bash
# Get status of a specific job
curl http://localhost:5002/job/<job_id>
```

### Redis CLI Access

```bash
# Access Redis directly
docker-compose exec redis redis-cli

# List all keys
KEYS *

# Check queue length
LLEN rq:queue:facial_recognition

# Monitor Redis operations
MONITOR
```

### Log Analysis

```bash
# Follow all relevant logs
docker-compose logs -f webhook-service rq-worker gpu-worker

# Filter for specific events
docker-compose logs webhook-service | grep "Processing event"
docker-compose logs rq-worker | grep "Processing.*upload"
```

## Configuration

### Environment Variables

#### Webhook Service
- `REDIS_URL` - Redis connection URL (default: `redis://redis:6379`)
- `MINIO_ENDPOINT` - MinIO endpoint (default: `minio:9000`)
- `MINIO_USER` - MinIO access key
- `MINIO_PASSWORD` - MinIO secret key

#### RQ Worker
- `REDIS_URL` - Redis connection URL
- `WORKER_NAME` - Unique worker name (default: `worker-{pid}`)
- `QUEUE_NAMES` - Comma-separated queue names (default: `facial_recognition`)
- `MINIO_ENDPOINT` - MinIO endpoint
- `GPU_WORKER_URL` - GPU worker endpoint (default: `http://gpu-worker:5001`)

### MinIO Notification Configuration

```bash
# View current webhook configuration
mc admin config get myminio/ notify_webhook:lookingglass

# Update webhook endpoint
mc admin config set myminio/ notify_webhook:lookingglass \
  endpoint="http://webhook-service:5002/webhook/minio" \
  queue_limit="100"

# Restart MinIO to apply changes
mc admin service restart myminio/
```

## Scaling

### Horizontal Scaling

```bash
# Scale RQ workers
docker-compose up -d --scale rq-worker=4

# Check worker instances
docker-compose ps rq-worker
```

### Queue Management

```bash
# Add additional queues if needed
# Update docker-compose.yml to include:
# environment:
#   - QUEUE_NAMES=facial_recognition,video_processing,cleanup
```

## Troubleshooting

### Common Issues

1. **Jobs not being queued**
   - Check MinIO bucket notifications: `mc event list myminio/images`
   - Verify webhook endpoint accessibility: `curl http://webhook-service:5002/health`
   - Check MinIO webhook configuration

2. **Jobs failing**
   - Check RQ worker logs: `docker-compose logs rq-worker`
   - Verify GPU worker accessibility: `curl http://gpu-worker:5001/health`
   - Check Redis connectivity

3. **Performance Issues**
   - Increase RQ worker replicas
   - Monitor queue depth: `curl http://localhost:5002/queue/stats`
   - Check resource usage: `docker stats`

### Reset and Cleanup

```bash
# Remove all bucket notifications
./scripts/setup-minio-notifications.sh clean

# Clear Redis queue
docker-compose exec redis redis-cli FLUSHALL

# Restart all services
docker-compose restart
```

## Migration from Direct Processing

The new RQ-based system maintains compatibility with existing crawlers and doesn't require changes to the upload process. The key differences:

1. **Before**: Upload → Direct processing call → Block until complete
2. **After**: Upload → Notification → Queue job → Async processing

Existing crawlers will continue to work without modification, but processing will now happen asynchronously in the background.

## Advanced Features

### Job Prioritization

```python
# High priority job
job = job_queue.enqueue(
    route_processing_job, 
    bucket_name, object_name, event_name,
    job_timeout='10m',
    priority=10  # Higher number = higher priority
)
```

### Job Scheduling

```python
from datetime import datetime, timedelta

# Schedule job for later
job = job_queue.enqueue_at(
    datetime.utcnow() + timedelta(minutes=5),
    route_processing_job,
    bucket_name, object_name, event_name
)
```

### Failed Job Handling

```python
# Jobs that fail are automatically moved to failed queue
# Check failed jobs via API
curl http://localhost:5002/queue/stats

# Retry failed jobs (can be implemented)
```

## API Reference

### Webhook Service Endpoints

#### POST /webhook/minio
Receives MinIO bucket notifications.

**Request**: MinIO S3 event notification JSON
**Response**: 
```json
{
  "status": "success",
  "processed_jobs": 1,
  "jobs": [
    {
      "job_id": "abc123",
      "bucket": "images",
      "object": "photo.jpg", 
      "event": "s3:ObjectCreated:Put",
      "status": "queued"
    }
  ]
}
```

#### GET /health
Health check and basic queue statistics.

#### GET /job/{job_id}
Get status of specific job.

#### GET /queue/stats  
Get detailed queue statistics.

#### POST /test/queue
Manually queue a test job.

This completes the RQ integration for LookingGlass! The system now provides asynchronous, scalable, and reliable processing of uploaded media files. 