# import os
# import numpy as np
# from PIL import Image
# from sklearn.model_selection import StratifiedKFold
# from sklearn.decomposition import PCA
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score
# import matplotlib.pyplot as plt

# # --- CONFIGURATION ---
# DATA_PATH = 'att_faces/'  # ต้องแน่ใจว่าโฟลเดอร์นี้มีอยู่
# IMAGE_SIZE = (92, 112)
# N_SPLITS = 5  # ทำ 5-Fold Cross-Validation
# K_NEIGHBORS = 1 # ใช้ 1-NN Classifier (เป็นมาตรฐานในงานวิจัย Eigenfaces)

# def load_orl_data(path):
#     """โหลดภาพทั้งหมดจาก ORL/ATT Database"""
#     X = []  # ข้อมูลภาพ
#     y = []  # ป้ายชื่อ (Labels)
#     target_names = []
    
#     # โฟลเดอร์ s1, s2, ..., s40 คือแต่ละคน
#     for i, person_dir in enumerate(sorted(os.listdir(path))):
#         person_path = os.path.join(path, person_dir)
#         if os.path.isdir(person_path):
#             person_label = i
#             target_names.append(person_dir)
            
#             # โหลดภาพ 10 ภาพของแต่ละคน
#             for filename in os.listdir(person_path):
#                 if filename.endswith('.pgm'):
#                     img_path = os.path.join(person_path, filename)
#                     # โหลดภาพ, แปลงเป็น Grayscale, และปรับขนาด
#                     img = Image.open(img_path).convert('L')
                    
#                     # แปลงภาพเป็น array และปรับเป็น 1D Vector
#                     img_vector = np.array(img, dtype='float64').flatten()
                    
#                     X.append(img_vector)
#                     y.append(person_label)
                    
#     return np.array(X), np.array(y), target_names

# def run_pca_experiment(X, y):
#     """
#     ทำการทดลองวิเคราะห์ความแม่นยำเทียบกับจำนวนมิติข้อมูล k
#     """
#     print(f"Total images: {X.shape[0]}, Total features (pixels): {X.shape[1]}")
    
#     # กำหนดค่า k ที่ต้องการทดสอบ (จำนวน Eigenfaces ที่เลือก)
#     # เราจะเลือก k ในช่วงที่หลากหลายและครอบคลุม
#     max_k = X.shape[0] - 1 # จำนวนมิติสูงสุดที่ PCA สามารถสร้างได้คือ N-1 (N=50)
#     k_values = [5, 10, 20, 30, 40]
    
#     # เพิ่มค่า k ที่เหมาะสมเพื่อให้เห็นจุด Peak ชัดเจน
#     if max_k > 40:
#         k_values.append(max_k)
#     else:
#          k_values.append(X.shape[0] - 2) # ใกล้เคียงค่าสูงสุด

#     k_values = sorted(list(set(k_values)))
    
#     accuracies = []
    
#     # ใช้ Stratified K-Fold เพื่อให้แน่ใจว่าแต่ละ Fold มีสัดส่วนของแต่ละคนเท่ากัน
#     skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
#     print(f"Testing k-values: {k_values} using {N_SPLITS}-Fold Cross-Validation...")

#     for k in k_values:
#         fold_accuracies = []
        
#         # 3.1 Loop ผ่านแต่ละ Fold ของ Cross-Validation
#         for train_index, test_index in skf.split(X, y):
#             X_train, X_test = X[train_index], X[test_index]
#             y_train, y_test = y[train_index], y[test_index]

#             # 3.2 ฝึก PCA บนชุดฝึกเท่านั้น
#             pca = PCA(n_components=k)
#             pca.fit(X_train)
            
#             # 3.3 ลดมิติข้อมูลชุดฝึกและชุดทดสอบ
#             X_train_pca = pca.transform(X_train)
#             X_test_pca = pca.transform(X_test)
            
#             # 3.4 ฝึก Classifier (k-NN)
#             clf = KNeighborsClassifier(n_neighbors=K_NEIGHBORS)
#             clf.fit(X_train_pca, y_train)
            
#             # 3.5 ประเมินผล
#             y_pred = clf.predict(X_test_pca)
#             accuracy = accuracy_score(y_test, y_pred)
#             fold_accuracies.append(accuracy)

#         # คำนวณความแม่นยำเฉลี่ยของ k นี้
#         avg_accuracy = np.mean(fold_accuracies)
#         accuracies.append(avg_accuracy)
#         print(f"k = {k}: Average Accuracy = {avg_accuracy:.4f}")

#     return k_values, accuracies

# def plot_results(k_values, accuracies):
#     """สร้างกราฟแสดงความสัมพันธ์ของ k กับ Accuracy"""
#     plt.figure(figsize=(10, 6))
#     plt.plot(k_values, accuracies, marker='o', linestyle='-', color='blue')
#     plt.title('Relationship between Dimensionality (k) and Face Recognition Accuracy')
#     plt.xlabel('Number of Principal Components (k)')
#     plt.ylabel('Average Recognition Accuracy (5-Fold CV)')
#     plt.grid(True)
    
#     # ไฮไลท์จุดที่มีความแม่นยำสูงสุด
#     max_acc = max(accuracies)
#     max_k = k_values[np.argmax(accuracies)]
#     plt.scatter(max_k, max_acc, color='red', s=100, label=f'Max Acc: {max_acc:.4f} at k={max_k}')
    
#     plt.legend()
#     plt.show()

# # --- RUN PROJECT ---
# if __name__ == "__main__":
#     try:
#         # 1. โหลดข้อมูล
#         X, y, target_names = load_orl_data(DATA_PATH)
        
#         # 2. ทำการทดลอง
#         k_values, accuracies = run_pca_experiment(X, y)
        
#         # 3. แสดงผล
#         plot_results(k_values, accuracies)
        
#     except FileNotFoundError:
#         print(f"ERROR: Cannot find the data folder at '{DATA_PATH}'")
#         print("Please ensure the 'att_faces' folder from ORL/ATT Database is in the same directory as this script.")

import os
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump
from config import ATT_FACES_PATH, IMAGE_SIZE, PCA_MODEL_PATH, CLASSIFIER_PATH, LABEL_NAMES_PATH, OPTIMAL_K, MODEL_DIR

def load_orl_data(path):
    """Loads all images from ORL/ATT Database and prepares data vectors."""
    X = []  # Image data (Flattened vectors)
    y = []  # Labels (Integer)
    target_names = [] # Folder names (s1, s2, ...)
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"ORL/ATT data not found at: {path}. Please place 'att_faces/' folder there.")

    print(f"Loading data from {path}...")
    for i, person_dir in enumerate(sorted(os.listdir(path))):
        person_path = os.path.join(path, person_dir)
        if os.path.isdir(person_path):
            person_label = i
            target_names.append(person_dir)
            
            for filename in os.listdir(person_path):
                if filename.endswith('.pgm'):
                    img_path = os.path.join(person_path, filename)
                    try:
                        # Open, convert to Grayscale ('L'), and flatten
                        img = Image.open(img_path).convert('L')
                        img_vector = np.array(img, dtype='float64').flatten()
                        X.append(img_vector)
                        y.append(person_label)
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
                    
    return np.array(X), np.array(y), target_names

def train_and_save_final_model(k):
    """Trains the final PCA and k-NN model using the optimal k and saves them."""
    X, y, target_names = load_orl_data(ATT_FACES_PATH)
    
    # 1. Train PCA (Eigenfaces)
    print(f"Training PCA with k={k} components...")
    pca = PCA(n_components=k)
    X_pca = pca.fit_transform(X)
    
    # 2. Train k-NN Classifier (on the PCA-transformed data)
    # Using the entire dataset (X) for training in this deployment phase for simplicity.
    # In a research context, you'd use a split dataset for a fairer deployment model.
    print("Training 1-NN Classifier...")
    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(X_pca, y)

    # 3. Save Artifacts
    dump(pca, PCA_MODEL_PATH)
    dump(clf, CLASSIFIER_PATH)
    dump(target_names, LABEL_NAMES_PATH)
    
    print("\n--- Training Complete ---")
    print(f"PCA and k-NN models saved to {MODEL_DIR}")
    print(f"Target Names (s1, s2, ...) saved. Model ready for use in app.py.")

if __name__ == "__main__":
    train_and_save_final_model(k=OPTIMAL_K)