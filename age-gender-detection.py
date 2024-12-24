from flask import Flask, jsonify, request
from flask_cors import CORS
import base64
import os
import io
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2 

app = Flask(__name__)
CORS(app)

# Load the pre-trained model (Ensure your model is in the same directory or provide the correct path)
#model = tf.keras.models.load_model('age.h5')

# Load OpenCV's pre-trained face detector (Haar Cascade or DNN can be used)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_face(image):
    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # If faces are detected, return the first face's bounding box
    if len(faces) == 0:
        return None, image  # No face detected, return original image
    
    # Get the coordinates of the first face
    x, y, w, h = faces[0]

    # Draw a rectangle around the face in the original image
    img_cv = np.array(image)  # Convert the PIL image to a NumPy array for OpenCV
    cv2.rectangle(img_cv, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw green rectangle
    
    # Convert back to PIL image to return it in the same format as input
    img_with_face = Image.fromarray(img_cv)

    # Crop the face from the image
    cropped_face = image.crop((x, y, x + w, y + h))  # Crop the face from the image

    return cropped_face, img_with_face  # Return both the cropped face and the image with the face rectangle


# Image preprocessing function (resize to fit model input)
def preprocess_image(image):
    image = image.resize((128, 128))  # Resize to model input size (128x128 as per your model's expected shape)
    image = np.array(image)  # Convert to numpy array
    image = image / 255.0  # Normalize the image to 0-1 range
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route('/get_info', methods=['POST'])
def get_info():
    data = request.get_json()

    if 'file' not in data:
        return jsonify({"error": "No file part"}), 400

    # Decode the base64 image
    img_data = base64.b64decode(data['file'].split(',')[1])  # Remove the 'data:image/jpeg;base64,' part
    image = Image.open(io.BytesIO(img_data))  # Open the image using PIL

    # Preprocess the image for prediction
    cropped_face, img_with_face = detect_face(image)


    if cropped_face is None:
        return jsonify({"error": "No face detected"}), 400
    
    processed_image = preprocess_image(cropped_face)

    # Make prediction
    prediction = model.predict(processed_image)

    print("Prediction:", prediction)

    # Extract the gender and age from the prediction
    gender_output = prediction[0][0]  # Assuming binary output (0 - female, 1 - male)
    age_output = prediction[1][0]  # Extract the predicted age

    # Convert the numpy arrays to Python native types (float) for JSON serialization
    gender_output = float(gender_output)
    age_output = float(age_output)

    # Determine the gender (0: female, 1: male)
    gender = 'male' if gender_output > 0.5 else 'female'

    # Convert the image with the rectangle to base64 for returning
    buffered = io.BytesIO()
    img_with_face.save(buffered, format="JPEG")
    img_with_face_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Prepare response data
    response = {
        "gender": gender,
        "age": int(age_output),
        "message": "Image processed successfully",
        "image_with_face": f"data:image/jpeg;base64,{img_with_face_base64}"
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
