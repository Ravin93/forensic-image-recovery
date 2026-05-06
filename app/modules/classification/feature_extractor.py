# app/modules/classification/feature_extractor.py

import numpy as np
import cv2


def extract_features(region):
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    variance = np.var(gray)
    mean = np.mean(gray)

    # gradient
    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1)
    gradient = np.mean(np.abs(sobel))

    # entropy
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    entropy = -np.sum(hist * np.log2(hist + 1e-7))

    return {
        "variance": float(variance),
        "mean": float(mean),
        "gradient": float(gradient),
        "entropy": float(entropy)
    }