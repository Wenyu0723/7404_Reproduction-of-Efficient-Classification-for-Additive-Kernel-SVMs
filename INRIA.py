import numpy as np
import cv2
import os
import re
import time
from sklearn.base import BaseEstimator, ClassifierMixin
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y, check_array

class Config:
    pos_img_dir = 'INRIAPerson/Train/pos'
    pos_ann_dir = 'INRIAPerson/Train/annotations'
    neg_img_dir = 'INRIAPerson/Train/neg'
    
    test_img_dir = 'INRIAPerson/Test/pos'
    test_ann_dir = 'INRIAPerson/Test/annotations'
    
    hog_params = {
        'orientations': 9,
        'pixels_per_cell': (8, 8),
        'cells_per_block': (2, 2),
        'block_norm': 'L2-Hys',
        'resize_to': (128, 64) 
    }
    
    neg_samples_per_img = 10
    scales = [0.6, 1.0, 1.3]         
    scales = [1.0]
    stride = 16                      
    nms_threshold = 0.1             
    score_threshold = -3          
    iou_threshold = 0.15             
    target_fppi = 2.0               
    distance_threshold = 40
    
    C = 1                       
    neg_ratio = 1.2                 

class INRIA_DataLoader:
    @staticmethod
    def parse_annotation(ann_path):

        #print(ann_path)
        with open(ann_path, 'r', errors='replace') as f:
            content = f.read()
        filename_match = re.search(r'Image filename : "(.+?)"', content)
        img_name = os.path.basename(filename_match.group(1).replace('\\', '/'))
        #print(img_name)
        if '.' not in img_name:
            img_name += '.png'
        boxes = []
        for x1,y1,x2,y2 in re.findall(r'\((\d+),\s*(\d+)\)\s*-\s*\((\d+),\s*(\d+)\)', content):
            if int(x1) < int(x2) and int(y1) < int(y2):
                boxes.append((int(x1), int(y1), int(x2), int(y2)))
        #print(f"boxes is {boxes}")
        return img_name, boxes
    @staticmethod
    def safe_resize(image, target_size):
        h, w = image.shape[:2]
        scale = max(target_size[0]/w, target_size[1]/h)
        resized = cv2.resize(image, None, fx=scale, fy=scale)

        y_start = max(0, (resized.shape[0] - target_size[1])//2)
        x_start = max(0, (resized.shape[1] - target_size[0])//2)
        return resized[y_start:y_start+target_size[1], x_start:x_start+target_size[0]]

    @classmethod
    def extract_hog(cls, image):
        resized = cls.safe_resize(image, Config.hog_params['resize_to'])
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        features = hog(gray, **{k:v for k,v in Config.hog_params.items() if k != 'resize_to'},
                      feature_vector=True)
        return features

    @classmethod
    def load_data(cls):
        X_pos = []
        for ann_file in os.listdir(Config.pos_ann_dir):
            ann_path = os.path.join(Config.pos_ann_dir, ann_file)
            try:
                filename, boxes = cls.parse_annotation(ann_path)
                img = cv2.imread(os.path.join(Config.pos_img_dir, filename))
                for x1,y1,x2,y2 in boxes:
                    roi = img[y1:y2, x1:x2]
                    #print(f"the roi is {roi}")
                    features = cls.extract_hog(roi)
                    #print(f"features we get is{features}")
                    X_pos.append(features)
            except Exception as e:
                print(f"Wrong: {e}")

        X_neg = []
        for img_file in os.listdir(Config.neg_img_dir):
            img = cv2.imread(os.path.join(Config.neg_img_dir, img_file))
            if img is None: continue
            h,w = img.shape[:2]
            for _ in range(Config.neg_samples_per_img):
                y = np.random.randint(0, h-128)
                x = np.random.randint(0, w-64)
                roi = img[y:y+128, x:x+64]
                X_neg.append(cls.extract_hog(roi))
        
        max_neg = int(len(X_pos)*Config.neg_ratio)
        if len(X_neg) > max_neg:
            X_neg = X_neg[:max_neg]
        y_pos=[1]*len(X_pos)
        y_neg=[-1]*len(X_neg)
        y = y_pos+y_neg
        print(f"the len of X_pos is {len(X_pos)} and the len of X_neg is {len(X_neg)}")
        return np.vstack(X_pos + X_neg), y
    
class HistogramIntersectionSVM(BaseEstimator, ClassifierMixin):
    
    def __init__(self, C=1.0):
        self.C = C
        self.scaler = StandardScaler()
        self.coef_ = None   
        self.intercept_ = 0.0
        self.dim_cache = [] 
    
    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=np.float64)
        X_scaled = self.scaler.fit_transform(X)
    
        from sklearn.svm import LinearSVC
        self.lsvm = LinearSVC(C=self.C, max_iter=10000, random_state=0)
        self.lsvm.fit(X_scaled, y)
        
        self.coef_ = self.lsvm.coef_.ravel()
        self.intercept_ = self.lsvm.intercept_[0]
        self._precompute_dimensions(X_scaled)
        return self
        
    def _precompute_dimensions(self, X):
        n_samples, n_dim = X.shape
        alpha_y = self.coef_
        
        self.dim_cache = []
        for d in range(n_dim):
            x_d = X[:, d]
            sorted_idx = np.argsort(x_d)
            sorted_x = x_d[sorted_idx]
            sorted_alpha = alpha_y[sorted_idx]
            
            A_r = np.cumsum(sorted_alpha * sorted_x)
            B_r = np.cumsum(sorted_alpha[::-1])[::-1] - sorted_alpha
            
            self.dim_cache.append( (sorted_x, A_r, B_r) )
    
    def decision_function(self, X):
        X = check_array(X)
        X_scaled = self.scaler.transform(X)
        scores = np.zeros(X_scaled.shape[0]) + self.intercept_
        
        for d in range(X_scaled.shape[1]):
            x_d = X_scaled[:, d]
            sorted_x, A_r, B_r = self.dim_cache[d]
            
            k = np.searchsorted(sorted_x, x_d, side='right') - 1
            k = np.clip(k, 0, len(sorted_x)-1)
            scores += A_r[k] + x_d * B_r[k]
        return scores
    
    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)
    
class Detector:
    def __init__(self, hog_params=None):
        self.hog_params = {
            'win_size': (64, 128),
            'block_size': (16, 16),
            'block_stride': (8, 8),
            'cell_size': (8, 8),
            'nbins': 9
        }
        if hog_params:  
            self.hog_params.update(hog_params)
        
        self.hog = cv2.HOGDescriptor(
            _winSize=self.hog_params['win_size'],
            _blockSize=self.hog_params['block_size'],
            _blockStride=self.hog_params['block_stride'],
            _cellSize=self.hog_params['cell_size'],
            _nbins=self.hog_params['nbins']
        )
    @staticmethod
    def sliding_window(image, window_size=(64, 128), step_size=4):
        for y in range(0, image.shape[0] - window_size[1] + 1, step_size):
            for x in range(0, image.shape[1] - window_size[0] + 1, step_size):
                yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

    def non_max_suppression(boxes, scores, overlapThresh=0.1):
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

    @classmethod
    def is_detection_correct(cls, detection, true_coords, threshold):
        dx, dy = detection[0], detection[1]
        for (tx, ty) in true_coords:
            if np.sqrt((dx - tx)**2 + (dy - ty)**2) < threshold:
                return True
        return False

    @classmethod
    def match_boxes(cls, pred_boxes, true_coords, threshold=50):
        #print(f"the pred_boxes are:{pred_boxes}")
        #print(f"the true_boxes are:{true_coords}")
        tp = 0
        fp = 0
        correct_image = False
        used_gt = [False] * len(true_coords)
        
        sorted_boxes = sorted(pred_boxes, key=lambda x: x[4], reverse=True)
        
        for box in sorted_boxes:
            dx, dy = box[0], box[1]
            matched = False
            
            for idx, (tx, ty) in enumerate(true_coords):
                if not used_gt[idx]:
                    distance = np.sqrt((dx - tx)**2 + (dy - ty)**2)
                    if distance < threshold:
                        tp += 1
                        used_gt[idx] = True
                        matched = True
                        correct_image = True
                        break
            if not matched:
                fp += 1
                
        return correct_image, tp, fp

    @classmethod
    def evaluate(cls, model):
        total_tp = 0
        total_fp = 0
        total_gt = 0
        correct_images = 0 
        start_time = time.time()
        detector = cls()
        test_files = [f for f in os.listdir(Config.test_img_dir) 
                     if f.lower().endswith(('.png','.jpg'))][:20]
        
        for idx, img_file in enumerate(test_files):
            img_path = os.path.join(Config.test_img_dir, img_file)
            ann_file = os.path.splitext(img_file)[0] + '.txt'
            ann_path = os.path.join(Config.test_ann_dir, ann_file)
            
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            _, gt_boxes = INRIA_DataLoader.parse_annotation(ann_path)
            true_coords = [(int(box[0]), int(box[1])) for box in gt_boxes]
            total_gt += len(true_coords)
            
            all_detections = []
            for scale in Config.scales:
                scaled_img = cv2.resize(img, None, fx=scale, fy=scale)
                gray_img = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2GRAY) 
                
                scaled_win_w = int(detector.hog_params['win_size'][0] * scale)
                scaled_win_h = int(detector.hog_params['win_size'][1] * scale)

                for (x, y, window) in cls.sliding_window(
                    gray_img, 
                    window_size=(scaled_win_w, scaled_win_h),  
                    step_size=Config.stride
                ):
                    try:
                        resized_win = cv2.resize(window, detector.hog_params['win_size'])
                        if resized_win.dtype != np.uint8:
                            resized_win = (resized_win * 255).astype(np.uint8)
                            
                        hog_feat = detector.hog.compute(resized_win).flatten()
                        score = model.decision_function(hog_feat.reshape(1,-1))[0]

                        if score > Config.score_threshold:
                        
                            orig_pts = (
                                int(x/scale), 
                                int(y/scale), 
                                int(scaled_win_w/scale), 
                                int(scaled_win_h/scale),
                                score
                            )
                            all_detections.append(orig_pts)
                    except Exception as e:
                        print(f"Window({x},{y})failed: {str(e)}")
                        continue
            
            final_boxes = cls.non_max_suppression(
                [det[:4] for det in all_detections],
                [det[4] for det in all_detections],
                Config.nms_threshold
            )[0]
            final_dets = [(*boxes, scores) for boxes, scores in zip(final_boxes, [det[4] for det in all_detections])]
            #print(final_dets)
            correct_image, tp, fp = cls.match_boxes(
                final_dets, 
                true_coords, 
                50
            )
            
            if correct_image:
                correct_images += 1
            total_tp += tp
            total_fp += fp
            

        fppi = total_fp / len(test_files)
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / total_gt if total_gt > 0 else 0
        accuracy = correct_images / len(test_files)
        
        return {
            'correct_images': correct_images,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'fppi': fppi,
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_gt': total_gt,
            'time_consumed': time.time() - start_time
        }

print("=== Training Start ===")
start_time = time.time()
X, y = INRIA_DataLoader.load_data()
print(f"Training set loaded successfully, time {time.time()-start_time:.1f}s")
print("Training IKSVM...")
model = HistogramIntersectionSVM(C=Config.C)
model.fit(X, y)
print(f"Training Complete, time {time.time()-start_time:.1f}s")
print("\n=== Start Evaluate ===")
eval_results = Detector.evaluate(model)
print("\nTest Result:")
print(eval_results)
print(f"2FPPI{Config.target_fppi}FPPI: {eval_results['recall']*100:.1f}%")
