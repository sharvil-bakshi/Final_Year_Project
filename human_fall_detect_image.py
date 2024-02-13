import cv2
import numpy as np
from keras.models import load_model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input



model = tf.keras.models.load_model(
    'model.h5',
    custom_objects=None,
    compile=False
)

# Load and preprocess the input image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64))  # Adjust the size based on your model's input requirements
    img = img / 255.0  # Normalize pixel values to the range [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def mark_and_save_image(image_path, output_path):
    original_img = cv2.imread(image_path)
    marked_img = original_img.copy()

    preprocessed_img = preprocess_image(image_path)
    prediction = model.predict(preprocessed_img)

    if prediction[0][0] > 0.5:
        # Fall detected: draw a red bounding box or any other indicator
        color = (0, 0, 255)  # Red color in BGR format
        cv2.rectangle(marked_img, (0, 0), (marked_img.shape[1], marked_img.shape[0]), color, thickness=2)
        cv2.putText(marked_img, 'Fall Detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Save the marked image
        cv2.imwrite(output_path, marked_img)
        print(f"Fall detected. Marked image saved to {output_path}")
    else:
        print("No fall detected.")

input_image_path = 'fall.jpg'
output_image_path = 'output_image_marked.jpg'
mark_and_save_image(input_image_path, output_image_path)
