# LookingGlass: Advanced Face Recognition System

LookingGlass is a production-grade distributed face recognition system that combines state-of-the-art computer vision techniques with efficient vector search capabilities. It's designed to process, analyze, and match faces across large datasets with high accuracy and performance.

It comes with a web crawler to collect images and videos of faces across the web, and a simple web app to upload images for recognition against the collected faces, providing a full, end-to-end setup, similar to PimEyes, or FaceCheck.

## 🌟 Key Features

### Production-Grade Face Detection with RetinaFace

-   State-of-the-art RetinaFace deep learning model for robust face detection
    -   Superior performance in challenging conditions (occlusions, angles, lighting)
    -   Built-in face alignment capabilities
    -   High-precision facial landmark detection (eyes, nose, mouth)
    -   Confidence scoring for each detection
    -   Handles multiple faces in crowded scenes
-   Multi-model detection pipeline:
    -   RetinaFace as primary detector
    -   HOG for fast preliminary screening
    -   CNN (GPU-accelerated) for verification
-   Smart detection fusion using IoU (Intersection over Union)
-   Confidence-based filtering (95%+ threshold)

### Advanced Image Processing

-   Automatic image enhancement pipeline:
    -   Adaptive histogram equalization for contrast improvement
    -   Intelligent noise reduction using fastNlMeans algorithm
    -   Format-aware processing (color/grayscale)
-   Quality assessment metrics:
    -   Brightness normalization
    -   Contrast evaluation
    -   Sharpness measurement
    -   Overall quality scoring

### State-of-the-Art Face Embeddings

-   Multi-model embedding generation:
    -   RetinaFace-aligned faces for optimal quality
    -   dlib's face_recognition (128-d embeddings)
    -   VGG-Face deep features (2622-d embeddings)
-   Combined 2750-dimensional feature vectors
-   L2-normalization for improved matching
-   GPU acceleration support

### Distributed Architecture

-   Three main services:
    1. **Crawler**: Intelligent web crawler for face collection
        - Proxy rotation with automatic failover
        - Recursive crawling with depth control
        - Image quality pre-filtering
        - High-Performance Video Processing:
            - Hardware-accelerated frame extraction
            - Adaptive frame skipping based on:
                - GPU utilization
                - Scene complexity
                - Face detection results
            - Batch processing for optimal GPU utilization
            - Parallel frame processing with CUDA acceleration
            - Smart keyframe selection using:
                - Scene change detection
                - Face pose variation
                - Quality metrics (blur, lighting, occlusion)
            - Support for various video platforms (YouTube, Vimeo, etc.)
            - Memory-efficient streaming processing
            - Temporal deduplication of faces
            - Processing speeds up to 500+ FPS on modern GPUs
    2. **Embedder**: Real-time face processing pipeline
        - Directory watching for instant processing
        - Parallel processing with ThreadPoolExecutor
        - Automatic GPU utilization when available
    3. **Lookup**: Fast face matching service
        - Web interface for easy querying
        - Real-time matching capabilities
        - Quality-aware result ranking

### Vector Search & Storage

-   Qdrant vector database integration
-   FAISS GPU acceleration for similarity search
-   Efficient metadata indexing
-   Cosine similarity matching
-   Payload indexing for fast filtering

### Production-Ready Features

-   Comprehensive error handling
-   Detailed logging system
-   Automatic deduplication
-   Resource cleanup
-   Parallel processing
-   GPU optimization
-   Docker containerization
-   Health monitoring

## 🚀 Performance

-   Face Detection: 95%+ accuracy on LFW benchmark
-   Real-time processing capabilities
-   Scalable to millions of faces
-   Sub-second query times
-   GPU acceleration for 10x+ speedup

## 🛠 Technical Stack

### Core Technologies

-   Python 3.6+
-   CUDA support
-   Docker & Docker Compose
-   GPU acceleration

### Machine Learning

-   TensorFlow 2.15.0
-   PyTorch 2.1.1
-   OpenCV 4.8.1
-   MTCNN
-   DeepFace
-   face_recognition (dlib)

### Infrastructure

-   Qdrant vector database
-   FAISS similarity search
-   Watchdog file monitoring
-   ThreadPoolExecutor parallelization

### Web & Networking

-   Flask web framework
-   Proxy rotation system
-   Selenium web crawler
-   Request handling

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/lookingglass.git
cd lookingglass

# Create a copy of the example environment file
cp .env.example .env

# Build and start the services
docker compose up --build
```

## 🔧 Configuration

### Environment Variables

-   `QDRANT_HOST`: Qdrant server hostname (default: "qdrant")
-   `QDRANT_PORT`: Qdrant server port (default: 6333)
-   `START_URL`: Starting URL for crawler
-   `MINIO_USER`: MinIO access key (default: "minioadmin")
-   `MINIO_PASSWORD`: MinIO secret key (default: "minioadmin")

### Hardware Requirements

-   Minimum 16GB RAM
-   NVIDIA GPU (recommended)
-   100GB storage space

## 📊 Architecture

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│   Crawler    │──▶│    Video     │──▶│   Embedder   │──▶│    Lookup    │
└──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
       │                 │                  │                  │
       ▼                 ▼                  ▼                  ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  Crawled     │   │Transcriptions│   │   Qdrant     │   │    Web UI    │
│   Faces      │   │              │   │  Database    │   │              │
└──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
```

## 🔍 Usage

### Web Interface

Access the lookup interface at `http://localhost:5000`

-   Upload images for face matching
-   View match results with confidence scores
-   Filter by quality metrics

### API Endpoints

-   `/lookup`: POST endpoint for face matching
-   Response includes:
    -   Match confidence scores
    -   Face locations
    -   Quality metrics
    -   Source information

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 References

-   [MTCNN Paper](https://arxiv.org/abs/1604.02878)
-   [VGG-Face](https://www.robots.ox.ac.uk/~vgg/software/vgg_face/)
-   [Qdrant Documentation](https://qdrant.tech/documentation/)
-   [FAISS](https://github.com/facebookresearch/faiss)
