import os
import numpy as np
from preprocessing import load_images_from_folder
from features import extract_all_features
from pca_reduction import apply_pca
from classifier import train_and_evaluate

DATASET_PATH = './dataset'  # Inside this folder: 'Tumor/', 'Healthy/' subfolders
IMAGE_SIZE = (128, 128)

print("Loading images...")
images, labels = load_images_from_folder(DATASET_PATH, IMAGE_SIZE)

print("Extracting features...")
X = np.array([extract_all_features(img) for img in images])
y = np.array(labels)

print("Applying PCA...")
X_reduced, pca = apply_pca(X)

print("Training and evaluating...")
accuracy, report = train_and_evaluate(X_reduced, y, model_type='svm')

print("Accuracy:", accuracy)
print("Classification Report:\n", report)
