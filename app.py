from flask import Flask, jsonify, request
from flask_cors import CORS
import base64
import os
import io
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import gdown

app = Flask(__name__)
CORS(app)


# Constants
GOOGLE_DRIVE_FILE_ID = '1slBW9pKYHLqpOU3pGeVpld0227n6GEws'  # Replace with your file ID
MODEL_PATH = 'age-1.h5'

# Download model if not present
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
        print("Model downloaded successfully.")

# Ensure the model is downloaded only once
download_model()

# Load the pre-trained model once when the app starts
model = tf.keras.models.load_model(MODEL_PATH)

# Load OpenCV's pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_face(image):
    """Detect face in the image and return cropped face and image with face rectangle."""
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        return None, image  # No face detected
    
    # Crop the first detected face
    x, y, w, h = faces[0]
    img_cv = np.array(image)  # Convert to NumPy array for OpenCV
    cv2.rectangle(img_cv, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangle
    
    img_with_face = Image.fromarray(img_cv)  # Convert back to PIL image
    cropped_face = image.crop((x, y, x + w, y + h))  # Crop the face from image
    
    return cropped_face, img_with_face

def preprocess_image(image):
    """Preprocess image to fit model input."""
    image = image.resize((128, 128))  # Resize to model input size (128x128)
    image = np.array(image) / 255.0  # Normalize to 0-1
    return np.expand_dims(image, axis=0)  # Add batch dimension

@app.route('/get_info', methods=['POST'])
def get_info():
    data = request.get_json()
    
    if 'file' not in data:
        return jsonify({"error": "No file part"}), 400

    # Decode base64 image
    img_data = base64.b64decode(data['file'].split(',')[1])
    image = Image.open(io.BytesIO(img_data))

    # Detect face
    cropped_face, img_with_face = detect_face(image)
    
    if cropped_face is None:
        return jsonify({"error": "No face detected"}), 400
    
    # Preprocess image and predict
    processed_image = preprocess_image(cropped_face)
    prediction = model.predict(processed_image)

    # Extract gender and age from prediction
    gender_output = prediction[0][0]
    age_output = prediction[1][0]

    # Determine gender
    gender = 'male' if gender_output > 0.5 else 'female'

    # Convert image with face rectangle to base64
    buffered = io.BytesIO()
    img_with_face.save(buffered, format="JPEG")
    img_with_face_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Return response
    response = {
        "gender": gender,
        "age": int(age_output),
        "message": "Image processed successfully",
        "image_with_face": f"data:image/jpeg;base64,{img_with_face_base64}"
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
