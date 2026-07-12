# app/modules/detection/segmentation.py

import cv2
import numpy as np


def segment_anomaly_map(anomaly_map, threshold=0.4):
    mask = (anomaly_map > threshold).astype(np.uint8) * 255

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask