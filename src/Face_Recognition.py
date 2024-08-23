import cv2
import dlib # type: ignore
import numpy as np

# Load the models
def load_models():
    detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
    return detector, shape_predictor, face_rec_model

# Extract face descriptors
def get_face_descriptor(image_path, detector, shape_predictor, face_rec_model):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    if len(faces) == 0:
        return None  # No faces detected

    face_descriptors = []
    for face in faces:
        shape = shape_predictor(gray, face)
        descriptor = face_rec_model.compute_face_descriptor(gray, shape)
        face_descriptors.append(np.array(descriptor))
    
    return face_descriptors

# Compare face descriptors
def compare_faces(reference_descriptors, target_descriptors, threshold=0.6):
    results = []
    for target_descriptor in target_descriptors:
        for reference_descriptor in reference_descriptors:
            distance = np.linalg.norm(reference_descriptor - target_descriptor)
            if distance < threshold:
                results.append(True)
            else:
                results.append(False)
    return any(results)

# Main function
def main():
    # Load models
    detector, shape_predictor, face_rec_model = load_models()

    # Load reference and target images
    reference_image_path = "reference_image.jpg"
    target_image_path = "target_image.jpg"

    # Get descriptors
    reference_descriptors = get_face_descriptor(reference_image_path, detector, shape_predictor, face_rec_model)
    if reference_descriptors is None:
        print("No faces detected in reference image.")
        return

    target_descriptors = get_face_descriptor(target_image_path, detector, shape_predictor, face_rec_model)
    if target_descriptors is None:
        print("No faces detected in target image.")
        return

    # Compare faces
    match_found = compare_faces(reference_descriptors, target_descriptors)
    if match_found:
        print("Face recognized!")
    else:
        print("Face not recognized.")

if __name__ == "__main__":
    main()
