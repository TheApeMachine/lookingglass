import os
import logging
import mimetypes
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import face_recognition
import deeplake
from minio import Minio
import base64
from redis import Redis
from rq import Queue
from jobs import route_processing_job, cleanup_processed_item, get_media_type
import threading

# Use environment variables for Redis connection
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
q = Queue(connection=Redis(host=REDIS_HOST, port=REDIS_PORT))

app = Flask(__name__)
CORS(app) # Enable CORS for development convenience

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

minio_client = Minio(
    os.getenv("MINIO_ENDPOINT", "localhost:9000"),
    access_key=os.getenv("MINIO_USER", "miniouser"),
    secret_key=os.getenv("MINIO_PASSWORD", "miniopassword"),
    secure=False
)

MINIO_BUCKET = os.getenv("MINIO_BUCKET", "scraped")
if not minio_client.bucket_exists(MINIO_BUCKET):
    minio_client.make_bucket(MINIO_BUCKET)

def listen_bucket_events(minio_client, bucket_name, q):
    """Listens for bucket notifications and enqueues jobs."""
    with minio_client.listen_bucket_notification(
        bucket_name,
        events=['s3:ObjectCreated:*', 's3:ObjectRemoved:*']
    ) as events:
        for event in events:
            for record in event.get('Records', []):
                try:
                    event_name = record['eventName']
                    bucket = record['s3']['bucket']['name']
                    object_name = record['s3']['object']['key']
                    
                    if event_name.startswith('s3:ObjectCreated:'):
                        media_type = get_media_type(object_name)
                        if media_type == 'page':
                            # page processing handled by graph service – skip
                            continue
                        q.enqueue(
                            route_processing_job, 
                            bucket_name=bucket, 
                            object_name=object_name, 
                            event_name=event_name, 
                            event_data=record
                        )
                    elif event_name.startswith('s3:ObjectRemoved:'):
                        q.enqueue(
                            cleanup_processed_item, 
                            bucket_name=bucket, 
                            object_name=object_name, 
                            event_name=event_name, 
                            event_data=record
                        )
                except KeyError as e:
                    logger.error(f"Malformed event record, missing key: {e} in record: {record}")


listener_thread = threading.Thread(
    target=listen_bucket_events,
    args=(minio_client, MINIO_BUCKET, q),
    daemon=True
)
listener_thread.start()

# Initialize DeepLake Dataset
# We use the MinIO bucket 'scraped' but store the DeepLake dataset in a sub-path 'faces_db'
DEEPLAKE_PATH = f"s3://{MINIO_BUCKET}/faces_db"
DEEPLAKE_CREDS = {
    "aws_access_key_id": os.getenv("MINIO_USER", "miniouser"),
    "aws_secret_access_key": os.getenv("MINIO_PASSWORD", "miniopassword"),
    "endpoint_url": f"http://{os.getenv('MINIO_ENDPOINT', 'localhost:9000')}",
    "s3_force_path_style": "true"
}

try:
    ds = deeplake.open(DEEPLAKE_PATH, creds=DEEPLAKE_CREDS)
    logger.info(f"Loaded existing DeepLake dataset at {DEEPLAKE_PATH}")
except Exception as e:
    logger.error(f"Failed to load DeepLake dataset at {DEEPLAKE_PATH}: {e}")

@app.route('/lookup', methods=['POST'])
def lookup():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']

    try:
        image = face_recognition.load_image_file(file)
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        if not face_encodings:
            return jsonify({'matches': [], 'faces_found': 0, 'face_locations': []})
        
        results = []
        for encoding in face_encodings:
            # DeepLake Vector Search using TQL
            try:
                # Convert encoding to comma-separated string for SQL-like query
                query_vec = ",".join(map(str, encoding))
                tql = f"SELECT * ORDER BY COSINE_SIMILARITY(embedding, ARRAY[{query_vec}]) DESC LIMIT 5"
                view = ds.query(tql)
                
                face_matches = []
                # Iterate through results (DeepLake views are subscriptable)
                for i in range(len(view)):
                    match_metadata = view['metadata'][i].data(as_numpy=False)
                    # Add score if available (TQL doesn't always return score explicitly in select * unless requested)
                    # but ordering works. We can assume descending relevance.
                    # Ideally we select COSINE_SIMILARITY(...) as score, but let's keep it simple first.
                    match = match_metadata
                    
                    try:
                        image_obj = minio_client.get_object(match['bucket'], match['object_name'])
                        image_bytes = image_obj.read()
                        encoded_img = base64.b64encode(image_bytes).decode('utf-8')
                        content_type, _ = mimetypes.guess_type(match['object_name'])
                        if not content_type or not content_type.startswith('image/'):
                            content_type = 'image/jpeg'
                        match['image_data_uri'] = f"data:{content_type};base64,{encoded_img}"
                    except Exception as e:
                        logger.error(f"Could not retrieve matched image from MinIO: {e}")
                        match['image_data_uri'] = None

                    face_matches.append(match)
                results.append(face_matches)
            except Exception as e:
                 logger.error(f"DeepLake query error: {e}")
                 results.append([])

        return jsonify({
            'matches': results, 
            'faces_found': len(face_encodings),
            'face_locations': face_locations
        })

    except Exception as e:
        logger.error(f"Lookup error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Flask app with lookup endpoint only...")
    app.run(host='0.0.0.0', port=5001) 