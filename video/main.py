"""Simple video processing with face detection and speech transcription."""

import os
import tempfile
from typing import Optional

import cv2
from moviepy.editor import VideoFileClip
from retinaface import RetinaFace
from speechbrain.inference import EncoderDecoderASR

# Preload the speech-to-text model from SpeechBrain
asr_model = EncoderDecoderASR.from_hparams(
    source="speechbrain/asr-conformer-transformerlm-librispeech",
    savedir="pretrained_models/asr-transformer-transformerlm-librispeech",
)


class VideoProcessor:
    """Process a video file with face detection and speech recognition."""

    def __init__(self, video_path: str, loader: Optional["MinioLoader"] = None, output_dir: str = "transcriptions"):
        self.video_path = video_path
        self.loader = loader
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _ensure_local_file(self) -> str:
        """Download the video via loader if it does not exist locally."""
        if os.path.exists(self.video_path):
            return self.video_path
        if self.loader is None:
            raise FileNotFoundError(self.video_path)
        data = self.loader.load()
        local_path = os.path.join("/tmp", os.path.basename(self.video_path))
        with open(local_path, "wb") as f:
            f.write(data)
        self.video_path = local_path
        return local_path

    def process_video(self) -> tuple[int, str]:
        """Run face detection and transcription on the video."""
        local_path = self._ensure_local_file()
        face_count = self.detect_faces(local_path)
        transcription = self.transcribe_video(local_path)
        self.save_transcription(transcription)
        return face_count, transcription

    def detect_faces(self, path: str) -> int:
        """Detect faces in video frames using RetinaFace.

        Returns the total number of faces detected across all sampled frames.
        """
        cap = cv2.VideoCapture(path)
        frame_idx = 0
        faces_total = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % 30 == 0:  # Sample every 30th frame
                temp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                cv2.imwrite(temp.name, frame)
                try:
                    faces = RetinaFace.detect_faces(temp.name)
                    if faces and isinstance(faces, dict):
                        faces_total += len(faces)
                finally:
                    os.unlink(temp.name)
            frame_idx += 1
        cap.release()
        return faces_total

    def transcribe_video(self, path: str) -> str:
        """Extract audio and run ASR on it."""
        with VideoFileClip(path) as clip:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                audio_path = tmp.name
            clip.audio.write_audiofile(audio_path, logger=None)
            text = asr_model.transcribe_file(audio_path)
            os.remove(audio_path)
        return text
        return text

    def save_transcription(self, transcription: str) -> None:
        """Save transcript text to the output directory."""
        name = os.path.splitext(os.path.basename(self.video_path))[0] + ".txt"
        out_path = os.path.join(self.output_dir, name)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(transcription)


class MinioLoader:
    """Load files from a MinIO bucket."""

    def __init__(self, bucket: str, object_name: str, endpoint: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None):
        self.bucket = bucket
        self.object_name = object_name
        endpoint = endpoint or os.getenv("MINIO_ENDPOINT", "minio:9000")
        user = user or os.getenv("MINIO_USER", "minioadmin")
        password = password or os.getenv("MINIO_PASSWORD", "minioadmin")
        from minio import Minio

        self.client = Minio(endpoint, access_key=user, secret_key=password, secure=False)

    def load(self) -> bytes:
        """Retrieve object data from MinIO."""
        response = self.client.get_object(self.bucket, self.object_name)
        data = response.read()
        response.close()
        response.release_conn()
        return data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process a video from disk or MinIO")
    parser.add_argument("video", help="Path to video file or object name in MinIO")
    parser.add_argument("--bucket", help="MinIO bucket name if loading from object storage")
    args = parser.parse_args()

    loader = MinioLoader(args.bucket, args.video) if args.bucket else None
    processor = VideoProcessor(args.video, loader=loader)
    faces, text = processor.process_video()
    print(f"Detected {faces} faces")
    print("Transcription saved.")
