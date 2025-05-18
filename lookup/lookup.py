import face_recognition
import faiss
import numpy as np
import sqlite3
import os
import pickle
from retinaface import RetinaFace

# --- Configuration ---
QUERY_IMAGE_PATH = "query_face.jpg" # Image containing the face(s) you want to identify
DB_PATH = "metadata.db"
FAISS_INDEX_PATH = "face_index.faiss"
MAPPING_PATH = "index_to_id.pkl"
NUM_NEIGHBORS_TO_FIND = 5 # How many potential matches to retrieve from Faiss
DISTANCE_THRESHOLD = 0.6 # Max L2 distance to consider a match (tune this value!)

# --- Check if necessary files exist ---
if not all(os.path.exists(p) for p in [QUERY_IMAGE_PATH, DB_PATH, FAISS_INDEX_PATH, MAPPING_PATH]):
    print("Error: One or more required files not found.")
    print("Please ensure the query image exists and the index/DB were built first.")
    exit()

# --- Load Faiss Index and Mapping ---
try:
    print(f"Loading Faiss index from {FAISS_INDEX_PATH}...")
    index = faiss.read_index(FAISS_INDEX_PATH)
    print(f"Loading index-to-personID mapping from {MAPPING_PATH}...")
    with open(MAPPING_PATH, 'rb') as f:
        index_to_person_id = pickle.load(f)
    print("Index and mapping loaded.")
except Exception as e:
    print(f"Error loading index or mapping: {e}")
    exit()

# --- Connect to Metadata DB ---
try:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    print("Connected to metadata database.")
except sqlite3.Error as e:
    print(f"Database connection error: {e}")
    exit()

# --- Process Query Image ---
print(f"\nProcessing query image: {QUERY_IMAGE_PATH}")
try:
    # Detect faces using RetinaFace
    faces = RetinaFace.detect_faces(QUERY_IMAGE_PATH)
    
    if not faces:
        print("No faces found in the query image.")
        conn.close()
        exit()

    print(f"Found {len(faces)} face(s) in the query image.")
    
    # Process each detected face
    for face_idx, (face_key, face_data) in enumerate(faces.items()):
        print(f"\n--- Processing Face #{face_idx + 1} ---")
        
        # Extract aligned face using RetinaFace
        aligned_face = RetinaFace.extract_faces(
            img_path=QUERY_IMAGE_PATH,
            align=True,
            face_idx=face_idx
        )[0]
        
        # Convert aligned face to format needed by face_recognition
        aligned_face = np.array(aligned_face)
        
        # Get face encoding
        query_encoding = face_recognition.face_encodings(aligned_face)[0]
        
        # Prepare query vector for Faiss
        query_vector = np.array([query_encoding]).astype('float32')
        
        # --- Perform Faiss Search ---
        distances, indices = index.search(query_vector, NUM_NEIGHBORS_TO_FIND)
        
        found_match = False
        print(f"  Face confidence score: {face_data['score']:.4f}")
        print("  Nearest neighbors (Index | Distance | Person ID | Name):")
        
        for j, faiss_index in enumerate(indices[0]):
            if faiss_index == -1:  # Faiss might return -1 if fewer than k neighbors exist
                continue
                
            distance = distances[0][j]
            
            # --- Map Faiss index back to person_id ---
            if faiss_index < len(index_to_person_id):
                person_id = index_to_person_id[faiss_index]
                
                # --- Retrieve metadata from DB ---
                cursor.execute("SELECT name FROM people WHERE person_id = ?", (person_id,))
                result = cursor.fetchone()
                name = result[0] if result else "Unknown ID"
                
                print(f"  - Neighbor {j+1}: Index={faiss_index}, Dist={distance:.4f}, ID={person_id}, Name='{name}'")
                
                # --- Check against threshold ---
                if distance <= DISTANCE_THRESHOLD:
                    print(f"      ----> Match found: '{name}' (Distance: {distance:.4f})")
                    found_match = True
            else:
                print(f"  - Neighbor {j+1}: Index={faiss_index} out of bounds for mapping.")
        
        if not found_match:
            print("  No match found within the distance threshold.")

except FileNotFoundError:
    print(f"Error: Query image file not found at '{QUERY_IMAGE_PATH}'.")
except Exception as e:
    print(f"An error occurred during query processing: {e}")
finally:
    # --- Cleanup ---
    conn.close()
    print("\nQuery process finished.")
