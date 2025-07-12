import os
import logging
import tempfile
import cv2
import hashlib
import numpy as np
from minio import Minio
import face_recognition
from retinaface import RetinaFace
from moviepy import VideoFileClip
from speechbrain.inference import EncoderDecoderASR
from qdrant_client import QdrantClient, models

# Add compatibility fixes for TensorFlow/Keras
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Fix for TensorFlow/Keras compatibility
import tensorflow as tf
if hasattr(tf, 'compat'):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

logger = logging.getLogger(__name__)

# Environment variables
MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', 'minio:9000')
MINIO_USER = os.getenv('MINIO_USER', 'miniouser')
MINIO_PASSWORD = os.getenv('MINIO_PASSWORD', 'miniopassword')
MINIO_BUCKET = os.getenv('MINIO_BUCKET', 'scraped')
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Initialize clients
minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_USER,
    secret_key=MINIO_PASSWORD,
    secure=False
)

# Ensure bucket exists
if not minio_client.bucket_exists(MINIO_BUCKET):
    minio_client.make_bucket(MINIO_BUCKET)
    logger.info(f"Created MinIO bucket: {MINIO_BUCKET}")

qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY, https=False)

# Load ASR model (lazy loading)
_asr_model = None

def get_asr_model():
    global _asr_model
    if _asr_model is None:
        try:
            logger.info("Loading ASR model...")
            _asr_model = EncoderDecoderASR.from_hparams(
                source="speechbrain/asr-conformer-transformerlm-librispeech",
                savedir="pretrained_models/asr-transformer-transformerlm-librispeech",
            )
            logger.info("ASR model loaded.")
        except Exception as e:
            logger.error(f"Failed to load ASR model: {str(e)}")
            _asr_model = None
    return _asr_model

def safe_detect_faces(image_rgb, retry_count=3):
    """
    Safely detect faces with error handling for TensorFlow/Keras compatibility issues.
    """
    for attempt in range(retry_count):
        try:
            faces = RetinaFace.detect_faces(image_rgb)
            return faces
        except Exception as e:
            error_msg = str(e).lower()
            if "padding" in error_msg and "valid" in error_msg:
                logger.warning(f"Padding compatibility issue detected (attempt {attempt + 1}): {e}")
                if attempt < retry_count - 1:
                    # Try to clear any cached models/sessions
                    try:
                        import gc
                        gc.collect()
                        if hasattr(tf, 'keras'):
                            tf.keras.backend.clear_session()
                    except:
                        pass
                    continue
            logger.error(f"Face detection failed after {attempt + 1} attempts: {e}")
            break
    return None

def process_image_upload(bucket_name, object_name, event_data=None):
    """
    RQ job function to process newly uploaded images for facial recognition.
    """
    try:
        logger.info(f"Processing image upload: {bucket_name}/{object_name}")
        
        # Download image from MinIO
        image_response = minio_client.get_object(bucket_name, object_name)
        image_bytes = image_response.read()
        
        # Get metadata directly from object
        try:
            stat_result = minio_client.stat_object(bucket_name, object_name)
            raw_metadata = stat_result.metadata or {}
            # Convert metadata keys to remove x-amz-meta- prefix and normalize to lowercase
            metadata = {}
            for key, value in raw_metadata.items():
                clean_key = key.lower().replace('x-amz-meta-', '')
                metadata[clean_key] = value
        except Exception as e:
            logger.info(f"No metadata found for {object_name}: {e}")
            metadata = {}
        
        # Process image with face recognition in-memory
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            logger.error(f"Could not decode image {object_name}")
            return {"status": "error", "error": f"Could not decode image {object_name}"}

        # Convert from BGR (OpenCV default) to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Use safe face detection with error handling
        faces = safe_detect_faces(image_rgb)

        if not isinstance(faces, dict) or faces is None:
            logger.info(f"No faces detected in {object_name}")
            return {"status": "success", "message": "No faces detected"}

        points_to_upsert = []
        for idx, (face_key, face_data) in enumerate(faces.items()):
            try:
                x1, y1, x2, y2 = face_data['facial_area']
                face_location = (int(y1), int(x2), int(y2), int(x1))
                face_encodings = face_recognition.face_encodings(image_rgb, [face_location])

                if face_encodings:
                    landmarks = {k: [float(v[0]), float(v[1])] for k, v in face_data['landmarks'].items()}
                    point = models.PointStruct(
                        id=hashlib.sha1(f"{object_name}_{idx}".encode()).hexdigest(),
                        vector=face_encodings[0].tolist(),
                        payload={
                            "bucket": bucket_name,
                            "object_name": object_name,
                            "source_url": metadata.get("source-url", ""),
                            "media_url": metadata.get("media-url", ""),
                            "face_location": [int(y1), int(x2), int(y2), int(x1)],
                            "confidence": float(face_data['score']),
                            "landmarks": landmarks,
                            "face_idx": idx,
                        }
                    )
                    points_to_upsert.append(point)
            except Exception as e:
                logger.warning(f"Failed to process face {idx} in {object_name}: {e}")
                continue
        
        if points_to_upsert:
            qdrant.upsert(collection_name='faces', points=points_to_upsert, wait=True)
            msg = f"Upserted {len(points_to_upsert)} faces from {object_name}"
            logger.info(msg)
            return {"status": "success", "result": msg}
        else:
            return {"status": "success", "message": "No face encodings generated"}

    except Exception as e:
        logger.error(f"Error processing image {object_name}: {str(e)}")
        return {"status": "error", "error": str(e)}
    finally:
        if 'image_response' in locals():
            image_response.close()
            image_response.release_conn()

def process_video_upload(bucket_name, object_name, event_data=None):
    """
    RQ job function to process newly uploaded videos for facial recognition.
    """
    try:
        logger.info(f"Processing video upload: {bucket_name}/{object_name}")
        
        # Download video from MinIO to a temporary file
        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(object_name)[1], delete=False) as tmp_video:
            minio_client.fget_object(bucket_name, object_name, tmp_video.name)
            video_path = tmp_video.name
        
        # 1. Detect faces
        cap = cv2.VideoCapture(video_path)
        faces_total = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps) if fps > 0 else 1
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: 
                break
            if frame_count % frame_interval == 0:
                try:
                    # Convert from BGR (OpenCV default) to RGB and process in-memory
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    faces = safe_detect_faces(frame_rgb)
                    if isinstance(faces, dict) and faces is not None:
                        faces_total += len(faces)
                except Exception as e:
                    logger.warning(f"Could not process a frame from {object_name}: {e}")

            frame_count += 1
        cap.release()

        # 2. Transcribe audio
        transcription = ""
        try:
            with VideoFileClip(video_path) as clip:
                if clip.audio:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
                        clip.audio.write_audiofile(tmp_audio.name, logger=None)
                        asr_model = get_asr_model()
                        if asr_model:
                            transcription = asr_model.transcribe_file(tmp_audio.name)
                        os.unlink(tmp_audio.name)
        except Exception as e:
            logger.warning(f"Failed to transcribe audio for {object_name}: {str(e)}")
        
        # Save transcription if we have one
        if transcription:
            transcription_path = f"/app/transcriptions/{os.path.splitext(object_name)[0]}.txt"
            os.makedirs(os.path.dirname(transcription_path), exist_ok=True)
            with open(transcription_path, "w") as f:
                f.write(transcription)

        result = {
            'face_count': faces_total,
            'transcription': transcription,
            'video_file': object_name
        }
        
        logger.info(f"Successfully processed video {object_name}: {faces_total} faces detected")
        return {"status": "success", "result": result}

    except Exception as e:
        logger.error(f"Error processing video {object_name}: {str(e)}")
        return {"status": "error", "error": str(e)}
    finally:
        if 'video_path' in locals() and os.path.exists(video_path):
            os.unlink(video_path)

def cleanup_processed_item(bucket_name, object_name, event_data=None):
    """
    RQ job function to handle cleanup when items are deleted from MinIO.
    This removes embeddings from the vector database.
    """
    try:
        logger.info(f"Cleaning up deleted item: {bucket_name}/{object_name}")
        
        # Remove faces from Qdrant that belong to this object
        # We'll filter by object_name in the payload
        try:
            qdrant.delete(
                collection_name='faces',
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="object_name",
                                match=models.MatchValue(value=object_name),
                            )
                        ]
                    )
                ),
                wait=True,
            )
            logger.info(f"Deleted face embeddings for {object_name}")

        except Exception as e:
            logger.warning(f"Could not clean up embeddings for {object_name}: {str(e)}")
        
        return {"status": "success", "message": f"Cleanup completed for {object_name}"}
        
    except Exception as e:
        logger.error(f"Error cleaning up {object_name}: {str(e)}")
        return {"status": "error", "error": str(e)}

def get_media_type(object_name):
    """
    Determine if the uploaded object is an image or video based on file extension.
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    video_extensions = ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.mkv']
    
    file_extension = os.path.splitext(object_name)[1].lower()
    
    if file_extension in image_extensions:
        return 'image'
    elif file_extension in video_extensions:
        return 'video'
    else:
        return 'unknown'

# Job routing function
def route_processing_job(bucket_name, object_name, event_name, event_data=None):
    """
    Route the processing job based on the event type and media type.
    This is the main entry point for processing jobs.
    """
    try:
        logger.info(f"Routing job for {event_name}: {bucket_name}/{object_name}")
        
        # Handle deletion events
        if 'ObjectRemoved' in event_name:
            return cleanup_processed_item(bucket_name, object_name, event_data)
        
        # Handle creation events
        if 'ObjectCreated' in event_name:
            # Route based on media type
            media_type = get_media_type(object_name)
            
            if media_type == 'image':
                return process_image_upload(bucket_name, object_name, event_data)
            elif media_type == 'video':
                return process_video_upload(bucket_name, object_name, event_data)
            else:
                logger.info(f"Skipping unsupported file type: {object_name}")
                return {"status": "skipped", "message": f"Unsupported file type: {media_type}"}
        
        return {"status": "error", "error": f"Unsupported event type: {event_name}"}
        
    except Exception as e:
        logger.error(f"Error routing job: {str(e)}")
        return {"status": "error", "error": str(e)} 