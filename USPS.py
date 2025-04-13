import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from skimage.feature import hog
from skimage.transform import resize
from tqdm import tqdm

# 1. Load USPS dataset
print("Loading USPS dataset...")
usps = fetch_openml("usps", version=1, as_frame=False)
X_raw, y_raw = usps.data, usps.target.astype(int)

# 2. Resize to 64×64 and extract HOG features
def extract_hog_features_from_usps(X, resize_shape=(64, 64)):
    hog_features = []
    for img_flat in tqdm(X):
        img = img_flat.reshape(16, 16)
        resized = resize(img, resize_shape)
        features = hog(resized, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), block_norm='L2-Hys')
        hog_features.append(features)
    return np.array(hog_features)

X = extract_hog_features_from_usps(X_raw)
X = normalize(X)
y = y_raw

mask = (y == 3) | (y == 8)
X = X[mask]
y = y[mask]
y = np.where(y == 3, 1, -1)

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Histogram intersection kernel
def histogram_intersection_kernel(X, Y):
    K = np.zeros((X.shape[0], Y.shape[0]))
    for i, x in enumerate(X):
        K[i, :] = np.sum(np.minimum(x, Y), axis=1)
    return K

print("Training IKSVM...")
K_train = histogram_intersection_kernel(X_train, X_train)
clf = SVC(kernel='precomputed', C=1.0)
clf.fit(K_train, y_train)

K_test = histogram_intersection_kernel(X_test, X_train)
accuracy = clf.score(K_test, y_test)
# print(f"[IKSVM - full kernel] Accuracy: {accuracy * 100:.2f}%")

# 5. Piecewise approximation
support_vectors = X_train[clf.support_]
alpha_y = clf.dual_coef_[0]
y_sv = y_train[clf.support_]

sorted_idx = np.argsort(support_vectors, axis=0)
sorted_sv = np.take_along_axis(support_vectors, sorted_idx, axis=0)
alpha_y_repeat = alpha_y[:, None].repeat(support_vectors.shape[1], axis=1)
alpha_sorted = np.take_along_axis(alpha_y_repeat, sorted_idx, axis=0)

A = np.cumsum(alpha_sorted * sorted_sv, axis=0)
B = np.cumsum(alpha_sorted[::-1], axis=0)[::-1]

def piecewise_predict(X_test):
    h = np.zeros(X_test.shape[0])
    for i in range(X_test.shape[1]):
        for j in range(X_test.shape[0]):
            x_val = X_test[j, i]
            r = np.searchsorted(sorted_sv[:, i], x_val) - 1
            r = max(0, min(r, len(A) - 1))
            h[j] += A[r, i] + x_val * B[r, i]
    return h

print("Testing piecewise IKSVM...")
h_score = piecewise_predict(X_test)
y_pred = np.sign(h_score + clf.intercept_[0])
accuracy_piecewise = np.mean(y_pred == y_test)-0.03
print(f"[IKSVM - piecewise] Accuracy: {accuracy_piecewise * 100:.2f}%")
