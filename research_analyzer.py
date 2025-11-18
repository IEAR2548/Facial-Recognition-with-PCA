import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

DATA_PATH = 'att_faces/'  
IMAGE_SIZE = (92, 112)
N_SPLITS = 5  
K_NEIGHBORS = 1 

def load_orl_data(path):
    """Loads all images from ORL/ATT Database."""
    X = []  # Image data (Flattened vectors)
    y = []  # Labels (Integer)
    target_names = []
    
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
                        img = Image.open(img_path).convert('L')
                        img_vector = np.array(img, dtype='float64').flatten()
                        X.append(img_vector)
                        y.append(person_label)
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
                    
    return np.array(X), np.array(y), target_names

def run_pca_experiment(X, y):
    """
    Performs Cross-Validation to analyze accuracy vs. k (dimensionality).
    """
    print(f"Total images: {X.shape[0]}, Total features (pixels): {X.shape[1]}")
    
    # Define k-values to test (Number of Eigenfaces)
    max_k = max_k_allowed = int(X.shape[0] * (N_SPLITS - 1) / N_SPLITS) # Maximum meaningful components (400-1 = 399 for full dataset)
    # Testing selected k-values for analysis
    k_values = [5, 10, 20, 30, 40, 50, 75, 100, 150, 200, 300, max_k]
    
    # Filter k_values to be less than or equal to the actual max_k
    k_values = [k for k in k_values if k <= max_k]
    k_values = sorted(list(set(k_values)))
    
    accuracies = []
    
    # Stratified K-Fold ensures each fold has representation from all people (classes)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
    print(f"Testing k-values: {k_values} using {N_SPLITS}-Fold Cross-Validation...")

    for k in k_values:
        fold_accuracies = []
        
        # Loop through each fold
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # 1. Train PCA on training set only
            pca = PCA(n_components=k)
            pca.fit(X_train)
            
            # 2. Transform both sets
            X_train_pca = pca.transform(X_train)
            X_test_pca = pca.transform(X_test)
            
            # 3. Train Classifier (1-NN)
            clf = KNeighborsClassifier(n_neighbors=K_NEIGHBORS)
            clf.fit(X_train_pca, y_train)
            
            # 4. Evaluate
            y_pred = clf.predict(X_test_pca)
            accuracy = accuracy_score(y_test, y_pred)
            fold_accuracies.append(accuracy)

        # Calculate the average accuracy for this k
        avg_accuracy = np.mean(fold_accuracies)
        accuracies.append(avg_accuracy)
        print(f"k = {k}: Average Accuracy = {avg_accuracy:.4f}")

    return k_values, accuracies

def plot_results(k_values, accuracies):
    """Creates a plot of k (dimensionality) vs. Accuracy."""
    plt.figure(figsize=(12, 7))
    plt.plot(k_values, accuracies, marker='o', linestyle='-', color='#007ACC', linewidth=2)
    
    max_acc = max(accuracies)
    max_k = k_values[np.argmax(accuracies)]
    
    # Highlight the peak
    plt.scatter(max_k, max_acc, color='red', s=150, zorder=5, label=f'Optimal k: {max_k} (Acc: {max_acc:.4f})')
    plt.axvline(x=max_k, color='red', linestyle='--', alpha=0.5)

    plt.title('Relationship between Dimensionality (k) and Face Recognition Accuracy (Eigenfaces)', fontsize=16)
    plt.xlabel('Number of Principal Components (k)', fontsize=14)
    plt.ylabel('Average Recognition Accuracy (5-Fold CV)', fontsize=14)
    plt.xticks(k_values, rotation=45)
    plt.yticks(np.arange(0, 1.05, 0.05))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

# --- RUN RESEARCH ---
if __name__ == "__main__":
    try:
        X, y, target_names = load_orl_data(DATA_PATH)
        
        # Run experiment (Cross-Validation)
        k_values, accuracies = run_pca_experiment(X, y)
        
        plot_results(k_values, accuracies)
        
    except FileNotFoundError:
        print(f"ERROR: Cannot find the data folder at '{DATA_PATH}'")
        print("Please ensure the 'att_faces' folder from ORL/ATT Database is in the same directory as this script.")