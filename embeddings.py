import cv2
import numpy as np #type:ignore
import psycopg2 #type:ignore
from scipy.spatial import distance #type:ignore
import dlib #type:ignore

# Database connection parameters
DB_HOST = 'localhost'
DB_NAME = 'Facial_Recognition_SDK'
DB_USER = 'postgres'
DB_PASS = 'password'
DB_PORT = 1000

def connect_db():
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        port=DB_PORT
    )
    return conn

def insert_embedding(person_id, angle, embedding):
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO face_embeddings (person_id, angle, embedding)
        VALUES (%s, %s, %s)
    """, (person_id, angle, embedding.tolist()))
    conn.commit()
    cur.close()
    conn.close()

def extract_features(image, detector, shape_predictor, face_rec_model):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    if len(faces) == 0:
        return None  # No faces detected

    features_list = []
    for face in faces:
        shape = shape_predictor(gray, face)
        descriptor = face_rec_model.compute_face_descriptor(image, shape)
        features_list.append(np.array(descriptor))
    
    return features_list

def load_models():
    SHAPE_PREDICTOR_PATH = "models/shape_predictor_68_face_landmarks.dat"
    FACE_REC_MODEL_PATH = "models/dlib_face_recognition_resnet_model_v1.dat"

    detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
    face_rec_model = dlib.face_recognition_model_v1(FACE_REC_MODEL_PATH)
    return detector, shape_predictor, face_rec_model

def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return None
    detector, shape_predictor, face_rec_model = load_models()
    features = extract_features(image, detector, shape_predictor, face_rec_model)
    return features

# Example usage
front_image_path = 'Path to front'
left_image_path = 'Path to left'
right_image_path = 'Path to right'

front_features = process_image(front_image_path)
left_features = process_image(left_image_path)
right_features = process_image(right_image_path)

# Insert embeddings into the database
person_id = 'Person Name'
if front_features:
    for feature in front_features:
        insert_embedding(person_id, 'front', feature)
if left_features:
    for feature in left_features:
        insert_embedding(person_id, 'left', feature)
if right_features:
    for feature in right_features:
        insert_embedding(person_id, 'right', feature)
