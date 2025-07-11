import os
import logging
import mimetypes
from flask import Flask, request, jsonify
import numpy as np
import face_recognition
from qdrant_client import QdrantClient, models
from minio import Minio
import base64
from redis import Redis
from rq import Queue
from jobs import route_processing_job, cleanup_processed_item
import threading

q = Queue(connection=Redis(host='redis', port=6379))
app = Flask(__name__)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

minio_client = Minio(
    os.getenv("MINIO_ENDPOINT", "minio:9000"),
    access_key=os.getenv("MINIO_USER", "minioadmin"),
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
            logger.info(f"Received raw event: {event}")
            
            for record in event.get('Records', []):
                try:
                    event_name = record['eventName']
                    bucket = record['s3']['bucket']['name']
                    object_name = record['s3']['object']['key']
                    
                    if event_name.startswith('s3:ObjectCreated:'):
                        logger.info(f"Enqueuing creation job for {bucket}/{object_name}")
                        q.enqueue(
                            route_processing_job, 
                            bucket_name=bucket, 
                            object_name=object_name, 
                            event_name=event_name, 
                            event_data=record
                        )
                    elif event_name.startswith('s3:ObjectRemoved:'):
                        logger.info(f"Enqueuing cleanup job for {bucket}/{object_name}")
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

QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY, https=False)

try:
    qdrant.get_collection(collection_name='faces')
    logger.info("Collection 'faces' already exists.")
except Exception:
    logger.info("Collection 'faces' not found, creating it.")
    qdrant.create_collection(
        collection_name='faces',
        vectors_config=models.VectorParams(size=128, distance=models.Distance.COSINE)
    )

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
            hits = qdrant.search(
                collection_name='faces',
                query_vector=encoding,
                limit=5,
                score_threshold=0.5
            )
            face_matches = []
            for hit in hits:
                match = hit.payload
                match['score'] = hit.score
                
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