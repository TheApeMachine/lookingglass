import os
import requests
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# The address of the new gpu-worker service
GPU_WORKER_URL = os.getenv("GPU_WORKER_URL", "http://gpu-worker:5001/lookup")

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

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
        # We don't need to save the file locally, we can send it in-memory
        file_data = file.read()
        
        # Forward the image to the gpu-worker service
        files = {'file': (file.filename, file_data, file.content_type)}
        response = requests.post(GPU_WORKER_URL, files=files, timeout=60)
        
        # Return the response from the gpu-worker directly to the client
        response.raise_for_status()
        return jsonify(response.json())

    except requests.exceptions.RequestException as e:
        # This could be a connection error, timeout, etc.
        return jsonify({'error': f"Could not connect to the processing service: {e}"}), 503
    except Exception as e:
        return jsonify({'error': f"An unexpected error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 
