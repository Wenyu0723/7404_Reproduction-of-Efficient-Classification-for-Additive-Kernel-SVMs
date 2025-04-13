import os
import glob
import cv2
import re
import numpy as np
import time
from skimage.feature import hog
from sklearn.svm import SVC
from PIL import Image

# ----------------------------
# 1. Data Loading & Feature Extraction
# ----------------------------

def load_pgm_image(file_path):
    img = Image.open(file_path).convert('L')
    return np.array(img)

def extract_hog_features(image, resize_shape=(64, 64)):
    image = cv2.resize(image, resize_shape)
    features = hog(image, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True)
    return features

def load_dataset_from_pattern(pattern, label, resize_shape=(64,64)):
    features = []
    labels = []
    file_list = glob.glob(pattern)
    for file in file_list:
        img = load_pgm_image(file)
        if img is None:
            continue
        feat = extract_hog_features(img, resize_shape)
        features.append(feat)
        labels.append(label)
    return np.array(features), np.array(labels)

# ----------------------------
# 2. Histogram Intersection Kernel + SVM Training
# ----------------------------

def histogram_intersection_kernel(X, Y):
    K = np.zeros((X.shape[0], Y.shape[0]))
    for i, x in enumerate(X):
        K[i, :] = np.sum(np.minimum(x, Y), axis=1)
    return K

def train_iks_vm(X_train, y_train):
    K_train = histogram_intersection_kernel(X_train, X_train)
    clf = SVC(kernel='precomputed', C=1.0)
    clf.fit(K_train, y_train)
    return clf

# ----------------------------
# 3. Piecewise IKSVM Classifier
# ----------------------------

class IKSVMClassifier:
    def __init__(self, support_vectors, weighted_alphas, bias):
        self.support_vectors = support_vectors
        self.weighted_alphas = weighted_alphas
        self.bias = bias
        self.m, self.n = support_vectors.shape

        self.breakpoints = []
        self.slopes = []
        self.intercepts = []

        for i in range(self.n):
            values = support_vectors[:, i]
            weights = self.weighted_alphas

            sorted_idx = np.argsort(values)
            x_sorted = values[sorted_idx]
            w_sorted = weights[sorted_idx]

            slopes_i = []
            intercepts_i = []
            for j in range(1, len(x_sorted)):
                x0, x1 = x_sorted[j - 1], x_sorted[j]
                if x1 == x0:
                    slope = 0.0
                else:
                    slope = (w_sorted[j - 1] * x0 + w_sorted[j] * x1) / 2 / (x1 - x0)
                intercept = 0
                slopes_i.append(slope)
                intercepts_i.append(intercept)

            self.breakpoints.append(x_sorted)
            self.slopes.append(slopes_i)
            self.intercepts.append(intercepts_i)

    def h_i(self, s, i):
        x = self.breakpoints[i]
        slopes = self.slopes[i]
        intercepts = self.intercepts[i]

        if s <= x[0]:
            return slopes[0] * (s - x[0]) + intercepts[0]
        elif s >= x[-1]:
            return slopes[-1] * (s - x[-2]) + intercepts[-1]

        for j in range(1, len(x)):
            if x[j-1] <= s <= x[j]:
                return slopes[j-1] * (s - x[j-1]) + intercepts[j-1]

        return 0.0

    def decision_function(self, x):
        total = 0.0
        for i in range(self.n):
            total += self.h_i(x[i], i)
        return total + self.bias

    def predict(self, X):
        results = np.array([self.decision_function(x) for x in X])
        return np.sign(results)

# ----------------------------
# 4. Sliding Window & Non-Maximum Suppression
# ----------------------------

def sliding_window(image, window_size, step_size):
    for y in range(0, image.shape[0] - window_size[1] + 1, step_size):
        for x in range(0, image.shape[1] - window_size[0] + 1, step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

def non_max_suppression(boxes, scores, overlapThresh=0.3):
    if len(boxes) == 0:
        return [], []
    boxes = np.array(boxes)
    scores = np.array(scores)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= overlapThresh)[0]
        order = order[inds + 1]
    return boxes[keep].tolist(), scores[keep].tolist()

# ----------------------------
# 5. Ground Truth Parsing
# ----------------------------

def parse_true_locations(file_path):
    true_dict = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split(':')
        if len(parts) < 2:
            continue
        index = parts[0].strip()
        matches = re.findall(r'\((\-?\d+),\s*(\-?\d+)\)', parts[1])
        coords = [(int(x), int(y)) for x, y in matches]
        true_dict[index] = coords
    return true_dict

def is_detection_correct(detection, true_coords, threshold=20):
    dx, dy = detection[0], detection[1]
    for (tx, ty) in true_coords:
        if np.sqrt((dx - tx)**2 + (dy - ty)**2) < threshold:
            return True
    return False

# ----------------------------
# 6. Main Logic
# ----------------------------

if __name__ == '__main__':
    train_dir = r'C:\Users\Wenyu\Desktop\MDASC\7404\project\CarData\TrainImages'
    pos_pattern = os.path.join(train_dir, r'pos-*.pgm')
    neg_pattern = os.path.join(train_dir, r'neg-*.pgm')

    print("Loading training data...")
    X_pos, y_pos = load_dataset_from_pattern(pos_pattern, label=1)
    X_neg, y_neg = load_dataset_from_pattern(neg_pattern, label=-1)
    X_train = np.vstack((X_pos, X_neg))
    y_train = np.concatenate((y_pos, y_neg))

    print("Positive samples:", X_pos.shape)
    print("Negative samples:", X_neg.shape)

    clf = train_iks_vm(X_train, y_train)
    support_vectors = clf.support_vectors_
    weighted_alphas = clf.dual_coef_.flatten()
    bias = clf.intercept_[0]
    fast_classifier = IKSVMClassifier(support_vectors, weighted_alphas, bias)

    test_dir = r'C:\Users\Wenyu\Desktop\MDASC\7404\project\CarData\TestImages'
    true_loc_file = r'C:\Users\Wenyu\Desktop\MDASC\7404\project\CarData\trueLocations.txt'

window_size = (64, 64)
step_size = 4

detections_all = {}
print("Running detection on test images...")
start_time = time.time()

test_files = sorted([f for f in os.listdir(test_dir) if f.lower().endswith('.pgm')])

for idx, file in enumerate(test_files):
    img_path = os.path.join(test_dir, file)
    image = load_pgm_image(img_path)
    detections = []
    scores = []
    for (x, y, window) in sliding_window(image, window_size, step_size):
        feat = extract_hog_features(window, resize_shape=window_size)
        score = fast_classifier.decision_function(feat)
        if score > -3:
            detections.append([x, y, window_size[0], window_size[1]])
            scores.append(score)
    boxes, final_scores = non_max_suppression(detections, scores, overlapThresh=0.1)
    detections_all[file] = (boxes, final_scores)

total_test_time = time.time() - start_time
avg_test_time = total_test_time / len(test_files)

true_dict = parse_true_locations(true_loc_file)
correct_images = 0
total_images = len(test_files)

for file in test_files:
    key = file.split('-')[1].split('.')[0].lstrip('0')
    key = key if key else '0'
    boxes, scores = detections_all.get(file, ([], []))
    true_coords = true_dict.get(key, [])
    correct_detected = False
    for detection in boxes:
        if is_detection_correct(detection, true_coords, threshold=50):
            correct_detected = True
            break
    if correct_detected:
        correct_images += 1

accuracy = correct_images / total_images * 100
print(f"Detection Accuracy: {accuracy:.2f}%")