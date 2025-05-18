import os
import time
import logging
import numpy as np
from retinaface import RetinaFace
import cv2
from PIL import Image
import face_recognition
from qdrant_client import QdrantClient
from qdrant_client.http import models
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from concurrent.futures import ThreadPoolExecutor

class ImageProcessor:
    def __init__(self):
        self.setup_logging()
        self.setup_qdrant()
        self.executor = ThreadPoolExecutor(max_workers=4)

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def setup_qdrant(self):
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", "6333"))
        
        self.qdrant = QdrantClient(host=host, port=port)
        
        # Check if collection exists, if not create it
        try:
            self.qdrant.get_collection('faces')
        except Exception:
            self.qdrant.create_collection(
                collection_name='faces',
                vectors_config=models.VectorParams(
                    size=128,  # face_recognition embedding size
                    distance=models.Distance.COSINE
                )
            )

    def process_image(self, image_path):
        """Process an image to extract face embeddings using RetinaFace and face_recognition."""
        try:
            # Check if image has already been processed
            image_hash = str(hash(image_path))
            search_result = self.qdrant.scroll(
                collection_name='faces',
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="image_path",
                            match=models.MatchValue(value=image_path)
                        )
                    ]
                ),
                limit=1
            )
            
            if search_result[0]:  # Image already processed
                self.logger.info(f"Image {image_path} already processed, skipping")
                return

            # Detect faces using RetinaFace
            faces = RetinaFace.detect_faces(image_path)
            
            if not faces:
                self.logger.warning(f"No faces detected in {image_path}")
                return

            # Load image for face_recognition
            image = face_recognition.load_image_file(image_path)
            
            # Process each face
            for face_idx, (face_key, face_data) in enumerate(faces.items()):
                try:
                    # Get face location from RetinaFace
                    x1, y1, x2, y2 = face_data['facial_area']
                    face_location = (int(y1), int(x2), int(y2), int(x1))  # Convert to face_recognition format
                    
                    # Extract aligned face using RetinaFace
                    aligned_face = RetinaFace.extract_faces(
                        img_path=image_path,
                        align=True,
                        face_idx=face_idx
                    )[0]
                    
                    # Convert aligned face to format needed by face_recognition
                    aligned_face = np.array(aligned_face)
                    
                    # Get face encoding
                    face_encoding = face_recognition.face_encodings(aligned_face)[0]
                    
                    # Store in Qdrant
                    self.qdrant.upsert(
                        collection_name='faces',
                        points=[
                            models.PointStruct(
                                id=hash(f"{image_path}_{face_idx}"),
                                vector=face_encoding.tolist(),
                                payload={
                                    "image_path": image_path,
                                    "face_location": face_location,
                                    "confidence": float(face_data['score']),
                                    "landmarks": face_data['landmarks'],
                                    "face_idx": face_idx,
                                    "processed_timestamp": time.time()
                                }
                            )
                        ]
                    )
                    
                    self.logger.info(f"Processed face {face_idx + 1} from {image_path}")
                
                except Exception as e:
                    self.logger.error(f"Error processing face {face_idx} in {image_path}: {e}")
                    continue

        except Exception as e:
            self.logger.error(f"Error processing image {image_path}: {e}")

class ImageHandler(FileSystemEventHandler):
    def __init__(self, processor):
        self.processor = processor

    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            self.processor.logger.info(f"New image detected: {event.src_path}")
            self.processor.executor.submit(self.processor.process_image, event.src_path)

def main():
    processor = ImageProcessor()
    handler = ImageHandler(processor)
    
    # Set up observers for both directories
    observers = []
    for directory in ['known_faces', 'crawled_faces']:
        if os.path.exists(directory):
            observer = Observer()
            observer.schedule(handler, directory, recursive=False)
            observer.start()
            observers.append(observer)
            processor.logger.info(f"Started watching directory: {directory}")
            
            # Process existing images
            for filename in os.listdir(directory):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(directory, filename)
                    processor.executor.submit(processor.process_image, image_path)
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        for observer in observers:
            observer.stop()
        processor.executor.shutdown()
    
    for observer in observers:
        observer.join()

if __name__ == "__main__":
    main() 
