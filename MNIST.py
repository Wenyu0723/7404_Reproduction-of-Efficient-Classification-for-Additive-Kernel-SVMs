import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from skimage.feature import hog

#Linear SVM
mnist = datasets.fetch_openml('mnist_784', version=1, parser='auto')
X, y = mnist.data.to_numpy(dtype=np.float32), mnist.target.astype(int).to_numpy()
x_train_hog=[]
print(X.shape)
for i in tqdm(range(len(X))):
    img=X[i].reshape(28,28)
    fd, hog_image = hog(img, orientations=8, pixels_per_cell=(4,4),cells_per_block=(1,1), visualize=True)
    x_train_hog.append(fd)
X=np.array(x_train_hog)
Y=np.array(y)
x_train,x_val,y_train,y_val=train_test_split(X,Y,test_size=0.2)

print(x_train.shape)
print(x_val.shape)
clf=SVC()
clf.fit(x_train,y_train)
res=clf.score(x_val,y_val)
print(f'Linear SVM Accuracy is {res}')

#IKSVM
# === Step 1: Load MNIST and Preprocess ===
print("Loading MNIST...")
mnist = datasets.fetch_openml('mnist_784', version=1, as_frame=False)
X = mnist['data'].reshape(-1, 28, 28).astype(np.uint8)
y = mnist['target'].astype(np.int32)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Step 2: HOG Feature Extraction with Overlapping Blocks ===
def extract_hog_features(images):
    features = []
    for img in tqdm(images, desc="Extracting HOG"):
        img_features = []
        for block_size in [28, 14, 7, 4]:
            step = block_size // 2  # overlapping blocks
            for i in range(0, 28 - block_size + 1, step):
                for j in range(0, 28 - block_size + 1, step):
                    patch = img[i:i+block_size, j:j+block_size]
                    feat = hog(patch,
                               orientations=12,
                               pixels_per_cell=(block_size, block_size),
                               cells_per_block=(1, 1),
                               visualize=False,
                               feature_vector=True)
                    img_features.extend(feat)
        features.append(img_features)
    return np.array(features)

print("Extracting HOG features...")
X_train_hog = extract_hog_features(X_train)
X_test_hog = extract_hog_features(X_test)

# Normalize features to [0,1]
scaler = MinMaxScaler()
X_train_hog = scaler.fit_transform(X_train_hog)
X_test_hog = scaler.transform(X_test_hog)

# === Step 3: Train SVM with Histogram Intersection Kernel ===
def histogram_intersection_kernel(X, Y):
    K = np.zeros((X.shape[0], Y.shape[0]))
    for i, x in enumerate(X):
        K[i, :] = np.sum(np.minimum(x, Y), axis=1)
    return K

#clf = SVC()
#clf.fit(K_train, y_train)
class_table_funcs = {}
class_bias = {}

for digit in range(10):
    print(f"\nTraining IKSVM for class {digit} vs rest...")
    y_binary = (y_train == digit).astype(int)
    K_train = histogram_intersection_kernel(X_train_hog, X_train_hog)
    clf = SVC(kernel='precomputed')
    clf.fit(K_train, y_binary)
    
    sv = X_train_hog[clf.support_]
    alpha = clf.dual_coef_[0]
    support_y = y_binary[clf.support_]
    coef = alpha * (2 * support_y - 1)

    h_funcs = {}
    for dim in range(sv.shape[1]):
        x_vals = sv[:, dim]
        weights = coef
        sorted_idx = np.argsort(x_vals)
        x_sorted = x_vals[sorted_idx]
        w_sorted = weights[sorted_idx]
        min_vals = []
        acc = 0
        last = -1
        for x_val, w in zip(x_sorted, w_sorted):
            if x_val != last:
                min_vals.append((x_val, acc))
                last = x_val
            acc += w * x_val
        if not min_vals:
            min_vals = [(0, 0)]
        x_pts, y_pts = zip(*min_vals)
        h_funcs[dim] = interp1d(x_pts, y_pts, bounds_error=False, fill_value=(y_pts[0], y_pts[-1]))
    
    class_table_funcs[digit] = h_funcs
    class_bias[digit] = clf.intercept_[0]
def predict_fast(X, table_funcs, bias_terms):
    preds = []
    for x in tqdm(X, desc="Fast Predicting"):
        scores = {}
        for cls in range(10):
            h = table_funcs[cls]
            score = sum(h[dim](x[dim]) for dim in range(len(x)))
            score += bias_terms[cls]
            scores[cls] = score
        preds.append(max(scores, key=scores.get))
    return np.array(preds)

y_pred = predict_fast(X_test_hog, class_table_funcs, class_bias)
acc = accuracy_score(y_test, y_pred)
print(f"\n Final IKSVM Accuracy: {acc * 100:.2f}%")
