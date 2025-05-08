from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2

# Import auth blueprint
from auth.routes import auth

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Register auth blueprint
app.register_blueprint(auth, url_prefix='/auth')

# Load model
try:
    model = load_model('models/fingerprint_cnn.h5')
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Class labels (update based on your training)
model_class_indices = {
    0: 'A+',
    1: 'A-',
    2: 'AB+',
    3: 'AB-',
    4: 'B+',
    5: 'B-',
    6: 'O+',
    7: 'O-'
}

# Helpers
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    return np.expand_dims(img, axis=(0, -1))  # Shape: (1, 128, 128, 1)

# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files.get('fingerprint')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                img_array = preprocess_image(filepath)
                prediction = model.predict(img_array)
                predicted_class = np.argmax(prediction)
                confidence = round(np.max(prediction) * 100, 2)
                result = model_class_indices.get(predicted_class, "Unknown")

                os.remove(filepath)

                return render_template('result.html', result=result, confidence=confidence)
            except Exception as e:
                flash(f'Error: {e}', 'danger')
                return redirect(url_for('predict'))
        else:
            flash('No file selected or invalid file type.', 'warning')
            return redirect(url_for('predict'))

    return render_template('predict.html')

@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)
