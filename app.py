import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from joblib import load
from PIL import Image
import cv2

from config import PCA_MODEL_PATH, CLASSIFIER_PATH, LABEL_NAMES_PATH, UPLOAD_FOLDER, ALLOWED_EXTENSIONS, IMAGE_SIZE 

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

pca = None
clf = None
target_names = None

def allowed_file(filename):
    """Checks if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_models():
    """Loads the pre-trained PCA and Classifier models."""
    global pca, clf, target_names
    try:
        pca = load(PCA_MODEL_PATH)
        clf = load(CLASSIFIER_PATH)
        target_names = load(LABEL_NAMES_PATH)
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Please ensure you have run 'model_trainer.py' first.")

def preprocess_image_for_recognition(filepath):
    """
    1. Converts image to Grayscale.
    2. Resizes the image to the standard size (92x112).
    3. Flattens it into a 1D vector.
    """
    try:
        img = Image.open(filepath).convert('L')
        
        img = img.resize(IMAGE_SIZE)
        
        img_vector = np.array(img, dtype='float64').flatten()
        
        return img_vector.reshape(1, -1)
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return None

def predict_face(img_vector):
    """Applies PCA transformation and predicts the label."""
    if pca is None or clf is None:
        return "ERROR: Model not loaded."

    img_pca = pca.transform(img_vector)

    predicted_label_index = clf.predict(img_pca)[0]

    predicted_name = target_names[predicted_label_index]
    
    return predicted_name

# --- FLASK ROUTES ---

@app.before_request
def check_model_loading():
    """Ensure models are loaded on first request."""
    if pca is None:
        load_models()

@app.route('/', methods=['GET'])
def index():
    """Renders the main upload page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles file upload, recognition, and result display."""
    if 'file' not in request.files:
        return render_template('index.html', error='ไม่พบไฟล์ในคำขอ.')
    
    file = request.files['file']
    
    if file.filename == '':
        return render_template('index.html', error='กรุณาเลือกไฟล์ภาพ.')
    
    if file and allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        img_vector = preprocess_image_for_recognition(filepath)
        
        if img_vector is None:
            return render_template('index.html', error='ไม่สามารถประมวลผลไฟล์ภาพได้. โปรดตรวจสอบว่าเป็นไฟล์ภาพที่ถูกต้อง.')
        
        prediction = predict_face(img_vector)

        return render_template('index.html', filename=filename, prediction=prediction)
    else:
        return render_template('index.html', error='รูปแบบไฟล์ไม่ถูกต้อง. อนุญาตเฉพาะ png, jpg, jpeg, pgm เท่านั้น.')

if __name__ == '__main__':
    load_models() 
    app.run(debug=True)