#!/bin/bash

# GPU Service Entrypoint Script
# This script helps with initialization and debugging

set -e

echo "=== GPU Service Initialization ==="
echo "Python version: $(python --version)"
echo "TensorFlow version: $(python -c 'import tensorflow as tf; print(tf.__version__)')"
echo "Keras version: $(python -c 'import keras; print(keras.__version__)')"
echo "NumPy version: $(python -c 'import numpy as np; print(np.__version__)')"

# Check GPU availability
echo "=== GPU Check ==="
python -c "
import tensorflow as tf
print('TensorFlow GPU available:', tf.config.list_physical_devices('GPU'))
print('CUDA available:', tf.test.is_built_with_cuda())
"

# Check if we can import key libraries
echo "=== Library Import Check ==="
python -c "
try:
    import face_recognition
    print('✓ face_recognition imported successfully')
except Exception as e:
    print('✗ face_recognition import failed:', e)

try:
    from retinaface import RetinaFace
    print('✓ RetinaFace imported successfully')
except Exception as e:
    print('✗ RetinaFace import failed:', e)

try:
    import cv2
    print('✓ OpenCV imported successfully')
except Exception as e:
    print('✗ OpenCV import failed:', e)

try:
    from qdrant_client import QdrantClient
    print('✓ Qdrant client imported successfully')
except Exception as e:
    print('✗ Qdrant client import failed:', e)
"

# Set environment variables for better stability
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "=== Starting Application ==="
echo "Command: $@"

# Execute the passed command
exec "$@" 