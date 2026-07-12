import cv2


def deblur(image):
    kernel = cv2.getGaussianKernel(5, 1)
    return cv2.filter2D(image, -1, kernel)