import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ATT_FACES_PATH = os.path.join(BASE_DIR, 'att_faces')
IMAGE_SIZE = (92, 112) # Standard ORL size

MODEL_DIR = os.path.join(BASE_DIR, 'model')
os.makedirs(MODEL_DIR, exist_ok=True) 

PCA_MODEL_PATH = os.path.join(MODEL_DIR, 'pca_model.joblib')
CLASSIFIER_PATH = os.path.join(MODEL_DIR, 'knn_classifier.joblib')
LABEL_NAMES_PATH = os.path.join(MODEL_DIR, 'label_names.joblib')

OPTIMAL_K = 40 

UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True) 

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pgm'}