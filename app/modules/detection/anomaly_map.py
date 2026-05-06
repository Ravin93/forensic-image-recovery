# app/modules/detection/anomaly_map.py

import cv2
import numpy as np


def compute_anomaly_map(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Variance locale
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    variance = cv2.absdiff(gray, blur)

    # Gradient
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    gradient = cv2.magnitude(sobelx, sobely)

    # Normalisation
    variance = cv2.normalize(variance, None, 0, 1, cv2.NORM_MINMAX)
    gradient = cv2.normalize(gradient, None, 0, 1, cv2.NORM_MINMAX)

    anomaly = (variance + gradient) / 2.0

    return anomaly