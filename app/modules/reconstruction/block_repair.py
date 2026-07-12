import cv2


def repair_blocks(image):
    return cv2.blur(image, (8, 8))