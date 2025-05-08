import os
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf

# Load your model
model = tf.keras.models.load_model('models/blood_group_cnn.h5')

# Folder path containing multiple test images
test_folder = r"C:\Users\jenif\Desktop\blood-group-predictor\test_images"

# Iterate through all images in the folder
for filename in os.listdir(test_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(test_folder, filename)

        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(150, 150), color_mode='rgb')
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)[0]
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        print(f"🖼️ Image: {filename}")
        print(f"🔍 Predicted Class: {predicted_class} | Confidence: {confidence:.2f}%\n")
