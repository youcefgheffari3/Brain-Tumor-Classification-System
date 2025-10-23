import cv2
import numpy as np
import os
from collections import Counter

def load_images_from_folder(folder, target_size=None):
    images = []
    labels = []
    for label_folder in os.listdir(folder):
        label_path = os.path.join(folder, label_folder)
        if os.path.isdir(label_path):
            for filename in os.listdir(label_path):
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    img_path = os.path.join(label_path, filename)
                    img = cv2.imread(img_path, 0)
                    if img is not None:
                        if target_size:
                            img = cv2.resize(img, target_size)
                        images.append(img)
                        labels.append(label_folder)
    return images, labels
