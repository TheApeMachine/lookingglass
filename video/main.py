from speechbrain.inference import EncoderDecoderASR

asr_model = EncoderDecoderASR.from_hparams(
    source="speechbrain/asr-conformer-transformerlm-librispeech", 
    savedir="pretrained_models/asr-transformer-transformerlm-librispeech"
)

class VideoProcessor:
    def __init__(self, video_path):
        self.video_path = video_path

    def process_video(self):
        pass

    def detect_faces(self):
        pass

    def transcribe_video(self):
        pass

    def save_transcription(self, transcription):
        pass

class MinioLoader: