import cv2
import numpy as np
from skimage.feature import local_binary_pattern, hog

def extract_lbp(img, P=8, R=1):
    lbp = local_binary_pattern(img, P, R, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

def extract_hog(img):
    features, _ = hog(img, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=True)
    return features

def extract_sift(img):
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    if des is None:
        return np.zeros(128)
    return np.mean(des, axis=0)

def extract_orb(img):
    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(img, None)
    if des is None:
        return np.zeros(32)
    return np.mean(des, axis=0)

def extract_all_features(img):
    h, w = img.shape
    regions = [
        img[0:h//2, 0:w//2],
        img[0:h//2, w//2:],
        img[h//2:, 0:w//2],
        img[h//2:, w//2:]
    ]
    final_vector = []
    for r in regions:
        vec = np.concatenate([
            extract_lbp(r),
            extract_hog(r),
            extract_sift(r),
            extract_orb(r)
        ])
        final_vector.append(vec)
    return np.concatenate(final_vector)
