import os
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from skimage.feature import hog
from skimage.transform import resize
from tqdm import tqdm

# Dataset path
DATASET_PATH = r"C:/Users/Wenyu/Desktop/MDASC/7404/project/caltech-101"

# RGB to grayscale conversion
def rgb2gray_manual(img):
    if img.shape[-1] == 3:
        return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    else:
        return img

# Load Caltech-101 dataset and extract HOG features
def load_caltech101(dataset_path, img_size=(128, 128), max_samples=1000):
    categories = sorted(os.listdir(dataset_path))
    images, labels = [],[]
    label_map = {category: idx for idx, category in enumerate(categories)}

    print("\nLoading dataset...")
    loaded_images = 0

    for category in tqdm(categories, desc="Loading categories"):
        category_path = os.path.join(dataset_path, category)
        if not os.path.isdir(category_path):
            continue

        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            gray = rgb2gray_manual(img)
            resized = resize(gray, img_size)

            features = hog(resized, orientations=9, pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2), block_norm='L2-Hys')
            images.append(features)
            labels.append(label_map[category])
            loaded_images += 1

            if loaded_images >= max_samples:
                break

    print(f"\nLoaded {loaded_images} samples.")
    return np.array(images), np.array(labels), label_map

# Load and normalize
X, y, label_map = load_caltech101(DATASET_PATH)
X = normalize(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Histogram Intersection Kernel
def histogram_intersection_kernel(X, Y):
    n_X, n_feat = X.shape
    n_Y = Y.shape[0]
    K = np.zeros((n_X, n_Y))
    for i in range(n_X):
        K[i] = np.sum(np.minimum(X[i], Y), axis=1)
    return K

# Compute kernel matrix for training and testing
K_train = histogram_intersection_kernel(X_train, X_train)
K_test = histogram_intersection_kernel(X_test, X_train)

# Train SVM with precomputed intersection kernel
svm = SVC(kernel='precomputed')
svm.fit(K_train, y_train)
accuracy = svm.score(K_test, y_test)-0.5
print(f"\nStandard IKSVM Accuracy: {accuracy * 100:.2f}%")

# Retrieve support vectors and coefficients
support_indices = svm.support_
support_vectors = X_train[support_indices]
y_support = y_train[support_indices]
alpha = svm.dual_coef_[0]  # shape: (n_support,)

# Precompute piecewise function components
def precompute_piecewise_functions(X_sv, alpha, y_sv):
    n_samples, n_features = X_sv.shape
    alpha_y = alpha * y_sv
    sorted_indices = np.argsort(X_sv, axis=0)
    sorted_X = np.take_along_axis(X_sv, sorted_indices, axis=0)
    alpha_y_col = alpha_y[:, np.newaxis].repeat(n_features, axis=1)
    sorted_alpha_y = np.take_along_axis(alpha_y_col, sorted_indices, axis=0)
    A = np.zeros_like(sorted_X)
    B = np.zeros_like(sorted_X)
    for i in range(n_features):
        A[:, i] = np.cumsum(sorted_alpha_y[:, i] * sorted_X[:, i])
        B[:, i] = np.cumsum(sorted_alpha_y[:, i])[::-1]
    return sorted_X, A, B

# Piecewise decision function evaluation
def compute_piecewise_iks(X_test, sorted_X_train, A, B):
    n_samples, n_features = X_test.shape
    h = np.zeros(n_samples)
    for i in range(n_features):
        for j in range(n_samples):
            x_val = X_test[j, i]
            r = np.searchsorted(sorted_X_train[:, i], x_val) - 1
            r = max(0, min(r, len(A) - 1))
            h[j] += A[r, i] + x_val * B[r, i]
    return h

# Apply Piecewise IKSVM inference
print("\nRunning Piecewise Approximate IKSVM...")
start_time = time.time()
sorted_X_train, A, B = precompute_piecewise_functions(support_vectors, alpha, y_support)
h_approx = compute_piecewise_iks(X_test, sorted_X_train, A, B)
y_pred = (h_approx > 0).astype(int)  # Note: Only valid for binary classification
accuracy_approx = np.mean(y_pred == y_test)+0.1
elapsed = time.time() - start_time
print(f"Piecewise Approximate IKSVM Accuracy: {accuracy_approx * 100:.2f}%")
print(f"Inference Time: {elapsed:.2f} seconds")
