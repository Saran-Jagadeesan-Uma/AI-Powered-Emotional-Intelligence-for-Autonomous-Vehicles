import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Define emotion dictionary
emotion_dict = {0: 'anger', 1: 'contempt', 2: 'disgust', 3: 'fear', 4: 'happy', 5: 'neutral', 6: 'sad', 7: 'surprise'}

# Load your trained model
model = load_model('emotion_detection_model.h5')

def predict_emotion(img_path):
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Image at {img_path} could not be loaded.")
        return None
    
    # Preprocess image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img = cv2.resize(img, (48, 48)) / 255.0  # Resize and normalize
    img = img.reshape(-1, 48, 48, 1)  # Reshape for CNN input
    
    # Make prediction
    prediction = model.predict(img)
    emotion = emotion_dict[np.argmax(prediction)]  # Get the emotion with the highest score
    
    return emotion

# Test with a new image
img_path = "Test Resources/happy.jpeg"
print(predict_emotion(img_path))
