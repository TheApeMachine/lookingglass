import os
import logging
import tempfile
import cv2
import hashlib
import numpy as np
from minio import Minio
import face_recognition
from retinaface import RetinaFace
from moviepy.editor import VideoFileClip
from speechbrain.inference import EncoderDecoderASR
from qdrant_client import QdrantClient, models

# Task queue
from redis import Redis
from rq import Queue

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

# Initialise Qdrant client
qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY, https=False)

# Ensure the 'faces' collection exists (vector size 128, cosine distance)
try:
    qdrant.get_collection(collection_name="faces")
except Exception:
    logger.info("Creating Qdrant collection 'faces'")
    qdrant.create_collection(
        collection_name="faces",
        vectors_config=models.VectorParams(size=128, distance=models.Distance.COSINE)
    )

# Initialise RQ queue so that jobs can be enqueued from this module
redis_conn = Redis(host=os.getenv("REDIS_HOST", "redis"), port=int(os.getenv("REDIS_PORT", 6379)))
q = Queue(connection=redis_conn)

# Load ASR model (lazy loading)
_asr_model = None

def get_asr_model():
    global _asr_model
    if _asr_model is None:
        try:
            _asr_model = EncoderDecoderASR.from_hparams(
                source="speechbrain/asr-conformer-transformerlm-librispeech",
                savedir="pretrained_models/asr-transformer-transformerlm-librispeech",
            )
        except Exception as e:
            logger.error(f"Failed to load ASR model: {str(e)}")
            _asr_model = None
    return _asr_model

def safe_detect_faces(image_bgr, retry_count=3):
    """
    Detect faces on a BGR (OpenCV-style) image using RetinaFace.
    The helper retries a few times to work around occasional TF/Keras padding errors.
    """
    import time, gc
    for attempt in range(retry_count):
        try:
            return RetinaFace.detect_faces(image_bgr)
        except Exception as e:
            error_msg = str(e).lower()

            # Known recoverable conditions
            recoverable = (
                "padding" in error_msg and "valid" in error_msg
            ) or (
                "cudnn" in error_msg or "cuda out of memory" in error_msg
            )

            logger.warning(
                f"safe_detect_faces(): attempt {attempt + 1}/{retry_count} failed: {e}."
            )

            if attempt < retry_count - 1 and recoverable:
                # Clear TF/Keras and Python garbage, wait exponential back-off then retry
                try:
                    gc.collect()
                    if hasattr(tf, "keras"):
                        tf.keras.backend.clear_session()
                except Exception:
                    pass

                time.sleep(0.5 * (2 ** attempt))
                continue

            # Not recoverable or out of retries
            logger.error("safe_detect_faces(): giving up after unrecoverable error")
            break
    return None

# =========================================
# Helper to extract faces from a BGR frame and upsert them
# =========================================


def _make_face_id(bucket: str, obj: str, frame_idx: int | None, face_idx: int) -> str:
    """Generate a deterministic 32-hex character ID for a face point."""
    tag = f"{bucket}:{obj}:{frame_idx if frame_idx is not None else 0}:{face_idx}"
    return hashlib.sha1(tag.encode("utf-8")).hexdigest()[:32]


def extract_and_upsert_faces(
    frame_bgr: np.ndarray,
    bucket_name: str,
    object_name: str,
    metadata: dict,
    frame_idx: int | None = None,
) -> int:
    """Detect faces, encode embeddings and upsert to Qdrant.

    Returns the number of faces upserted from this frame.
    """

    faces = safe_detect_faces(frame_bgr)
    if not faces or not isinstance(faces, dict):
        return 0

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    points: list[models.PointStruct] = []

    for idx, (face_key, face_data) in enumerate(faces.items()):
        try:
            x1, y1, x2, y2 = face_data["facial_area"]
            face_location = (int(y1), int(x2), int(y2), int(x1))
            enc = face_recognition.face_encodings(frame_rgb, [face_location])
            if not enc:
                continue

            landmarks = {
                k: [float(v[0]), float(v[1])] for k, v in face_data["landmarks"].items()
            }

            point = models.PointStruct(
                id=_make_face_id(bucket_name, object_name, frame_idx, idx),
                vector=enc[0].tolist(),
                payload={
                    "bucket": bucket_name,
                    "object_name": object_name,
                    "source_url": metadata.get("source-url", ""),
                    "media_url": metadata.get("media-url", ""),
                    "frame_idx": frame_idx,
                    "face_location": [int(y1), int(x2), int(y2), int(x1)],
                    "confidence": float(face_data["score"]),
                    "landmarks": landmarks,
                    "face_idx": idx,
                },
            )
            points.append(point)
        except Exception as e:
            logger.warning(f"Failed to encode face {idx} in {object_name}: {e}")

    if points:
        qdrant.upsert(collection_name="faces", points=points, wait=True)
    return len(points)

def process_image_upload(bucket_name, object_name, event_data=None):
    """
    RQ job function to process newly uploaded images for facial recognition.
    """
    try:
        image_response = minio_client.get_object(bucket_name, object_name)
        image_bytes = image_response.read()
        
        try:
            stat_result = minio_client.stat_object(bucket_name, object_name)
            raw_metadata = stat_result.metadata or {}
            # Convert metadata keys to remove x-amz-meta- prefix and normalize to lowercase
            metadata = {}
            for key, value in raw_metadata.items():
                clean_key = key.lower().replace('x-amz-meta-', '')
                metadata[clean_key] = value
        except Exception as e:
            logger.warning(f"No metadata found for {object_name}: {e}")
            metadata = {}
        
        # Process image with face recognition in-memory
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            logger.error(f"Could not decode image {object_name}")
            return {"status": "error", "error": f"Could not decode image {object_name}"}

        # Resize large images to prevent GPU memory issues
        max_dim = 1024
        h, w, _ = image.shape
        if h > max_dim or w > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h))

        upserted = extract_and_upsert_faces(image, bucket_name, object_name, metadata)
        if upserted:
            return {"status": "success", "result": f"Upserted {upserted} faces"}
        return {"status": "success", "message": "No faces detected"}

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
        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(object_name)[1], delete=False) as tmp_video:
            minio_client.fget_object(bucket_name, object_name, tmp_video.name)
            video_path = tmp_video.name
        
        # 1. Detect faces
        try:
            video_clip = VideoFileClip(video_path)
            
            # Extract and process audio
            try:
                audio_path = video_path + ".wav"
                video_clip.audio.write_audiofile(audio_path, codec="pcm_s16le")

                # Upload audio file to MinIO so workers can access it
                audio_key = os.path.basename(audio_path)
                minio_client.fput_object(bucket_name, audio_key, audio_path)

                # Enqueue audio processing job with key stored in MinIO
                q.enqueue(
                    process_audio_upload,
                    bucket_name=bucket_name,
                    object_name=audio_key,
                    event_data={"source_video": object_name}
                )
            except Exception as e:
                logger.error(f"Failed to extract or enqueue audio for {object_name}: {e}")
            
            # Process video frames for faces
            faces_total = 0
            # Use iter_frames for memory efficiency
            for frame_idx, frame in enumerate(video_clip.iter_frames(fps=1, dtype='uint8')):
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                faces_total += extract_and_upsert_faces(
                    frame_bgr,
                    bucket_name=bucket_name,
                    object_name=object_name,
                    metadata={},
                    frame_idx=frame_idx,
                )
            
            logger.info(f"Found {faces_total} faces in {object_name}")

        except Exception as e:
            logger.error(f"Error processing video {object_name} with MoviePy: {e}")
            # Fallback to OpenCV if MoviePy fails
            try:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    logger.error(f"Failed to open video {object_name} with OpenCV as well.")
                    return {"status": "error", "error": "Failed to open video file"}

                faces_total = 0
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_interval = int(fps) if fps > 0 else 1
                frame_count = 0
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if frame_count % frame_interval == 0:
                        faces_total += extract_and_upsert_faces(
                            frame,
                            bucket_name=bucket_name,
                            object_name=object_name,
                            metadata={},
                            frame_idx=frame_count,
                        )
                    
                    frame_count += 1
                
                logger.info(f"Found {faces_total} faces in {object_name} using OpenCV fallback.")
                cap.release()
            except Exception as fallback_e:
                logger.error(f"OpenCV fallback failed for {object_name}: {fallback_e}")
        
        finally:
            if 'video_clip' in locals():
                video_clip.close()
            # Clean up temporary files
            if os.path.exists(video_path):
                os.remove(video_path)
            if 'audio_path' in locals() and os.path.exists(audio_path):
                os.remove(audio_path)

        return {"status": "success", "message": f"Processed video {object_name}."}

    except Exception as e:
        logger.error(f"Error processing video {object_name}: {e}")
        return {"status": "error", "error": str(e)}

def process_audio_upload(bucket_name, object_name, event_data=None):
    """
    RQ job function to process audio files for transcription.
    """
    try:
        # Ensure transcription directory exists
        os.makedirs("/app/transcriptions", exist_ok=True)

        # Download audio from MinIO
        audio_obj = minio_client.get_object(bucket_name, object_name)
        audio_bytes = audio_obj.read()
        audio_path = tempfile.mktemp(suffix=os.path.splitext(object_name)[1])
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)

        asr_model = get_asr_model()
        if not asr_model:
            raise Exception("ASR model is not available.")
            
        logger.info(f"Transcribing {audio_path}...")
        text = asr_model.transcribe_file(audio_path)
        
        # Save transcription to a file
        transcription_filename = os.path.splitext(os.path.basename(object_name))[0] + ".txt"
        transcription_path = os.path.join("/app/transcriptions", transcription_filename)
        with open(transcription_path, "w") as f:
            f.write(text)
        
        logger.info(f"Transcription for {object_name} saved to {transcription_path}")
        
        return {"status": "success", "transcription": text}

    except Exception as e:
        logger.error(f"Error processing audio {object_name}: {e}")
        return {"status": "error", "error": str(e)}
    finally:
        # Clean up temp audio file if it was downloaded
        if 'audio_path' in locals() and os.path.exists(audio_path):
            os.remove(audio_path)

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
    html_extensions = ['.html', '.htm']
    
    file_extension = os.path.splitext(object_name)[1].lower()
    
    if file_extension in image_extensions:
        return 'image'
    elif file_extension in video_extensions:
        return 'video'
    elif file_extension in html_extensions:
        return 'page'
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