import os
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import face_recognition
import faiss
import numpy as np
import sqlite3
import pickle

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configuration
DB_PATH = "data/metadata.db"
FAISS_INDEX_PATH = "data/face_index.faiss"
MAPPING_PATH = "data/index_to_id.pkl"
NUM_NEIGHBORS_TO_FIND = 5
DISTANCE_THRESHOLD = 0.6

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

def load_resources():
    """Load Faiss index and mapping"""
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(MAPPING_PATH, 'rb') as f:
        index_to_person_id = pickle.load(f)
    return index, index_to_person_id

def get_person_info(person_id):
    """Get person info from database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM people WHERE person_id = ?", (person_id,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else "Unknown"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/lookup', methods=['POST'])
def lookup():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process image
        image = face_recognition.load_image_file(filepath)
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        if not face_encodings:
            return jsonify({'error': 'No faces found in image'}), 400

        # Load resources
        index, index_to_person_id = load_resources()
        
        results = []
        for face_encoding in face_encodings:
            # Prepare query vector
            query_vector = np.array([face_encoding]).astype('float32')
            
            # Perform search
            distances, indices = index.search(query_vector, NUM_NEIGHBORS_TO_FIND)
            
            face_matches = []
            for j, faiss_index in enumerate(indices[0]):
                if faiss_index == -1:
                    continue
                    
                distance = distances[0][j]
                if distance <= DISTANCE_THRESHOLD and faiss_index < len(index_to_person_id):
                    person_id = index_to_person_id[faiss_index]
                    name = get_person_info(person_id)
                    face_matches.append({
                        'name': name,
                        'confidence': float(1 - distance/2)  # Convert distance to confidence score
                    })
            
            if face_matches:
                results.append(face_matches)

        # Clean up
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'faces_found': len(face_encodings),
            'matches': results
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 
