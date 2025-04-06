import face_recognition
import faiss
import numpy as np
import sqlite3
import os
import pickle
from PIL import Image

# --- Configuration ---
KNOWN_FACES_DIR = "known_faces"  # Folder with images of known people (one face per image ideally)
DB_PATH = "metadata.db"
FAISS_INDEX_PATH = "face_index.faiss"
MAPPING_PATH = "index_to_id.pkl"
FACE_EMBEDDING_DIM = 128 # face_recognition uses 128-d embeddings

# --- Ensure known faces directory exists ---
if not os.path.exists(KNOWN_FACES_DIR):
    print(f"Error: Directory '{KNOWN_FACES_DIR}' not found.")
    print("Please create it and add images of known people.")
    exit()

# --- Setup SQLite Database ---
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS people (
    person_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    image_path TEXT
)""")
conn.commit()

# --- Process Known Faces ---
all_embeddings = []
index_to_person_id = [] # This list maps Faiss index (row number) to person_id

print(f"Processing images in '{KNOWN_FACES_DIR}'...")

for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(KNOWN_FACES_DIR, filename)
        person_name = os.path.splitext(filename)[0] # Use filename as name (adjust as needed)

        print(f"Processing {filename} for '{person_name}'...")

        try:
            image = face_recognition.load_image_file(image_path)
            # Find *all* faces, but we'll use the first for simplicity in this PoC
            # For production, you might handle multiple faces or ensure one face per file
            face_locations = face_recognition.face_locations(image)
            face_encodings = face_recognition.face_encodings(image, face_locations)

            if len(face_encodings) == 1:
                embedding = face_encodings[0]

                # --- Store metadata in SQLite ---
                try:
                    cursor.execute("INSERT OR IGNORE INTO people (name, image_path) VALUES (?, ?)",
                                   (person_name, image_path))
                    conn.commit()
                    # Get the person_id (either newly inserted or existing)
                    cursor.execute("SELECT person_id FROM people WHERE name = ?", (person_name,))
                    result = cursor.fetchone()
                    if result:
                        person_id = result[0]

                        # --- Store embedding and mapping ---
                        all_embeddings.append(embedding)
                        index_to_person_id.append(person_id) # Map this embedding's index to the person_id
                    else:
                        print(f"  Warning: Could not retrieve person_id for '{person_name}'. Skipping.")

                except sqlite3.Error as e:
                    print(f"  Database error for {person_name}: {e}")

            elif len(face_encodings) == 0:
                print(f"  Warning: No face found in {filename}. Skipping.")
            else:
                print(f"  Warning: Multiple faces found in {filename}. Skipping (for this simple PoC).")

        except Exception as e:
            print(f"  Error processing file {filename}: {e}")

# --- Check if we have any embeddings ---
if not all_embeddings:
    print("\nError: No face embeddings were successfully extracted. Cannot build index.")
    conn.close()
    exit()

# --- Convert embeddings to NumPy array ---
embeddings_np = np.array(all_embeddings).astype('float32') # Faiss requires float32
print(f"\nTotal embeddings collected: {embeddings_np.shape[0]}")

# --- Build Faiss Index ---
# IndexFlatL2 performs exact search (calculates L2 distance to all items).
# Good for accuracy/simplicity, but slower for huge datasets than ANN indexes.
# Alternatives for larger scale: faiss.IndexIVFFlat, faiss.IndexHNSWFlat
print("Building Faiss index (IndexFlatL2)...")
index = faiss.IndexFlatL2(FACE_EMBEDDING_DIM)
index.add(embeddings_np) # Add the embeddings to the index
print(f"Faiss index built. Total vectors in index: {index.ntotal}")

# --- Save the Index and Mapping ---
print(f"Saving Faiss index to {FAISS_INDEX_PATH}...")
faiss.write_index(index, FAISS_INDEX_PATH)

print(f"Saving index-to-personID mapping to {MAPPING_PATH}...")
with open(MAPPING_PATH, 'wb') as f:
    pickle.dump(index_to_person_id, f)

# --- Cleanup ---
conn.close()
print("\nDatabase and index build process complete.")