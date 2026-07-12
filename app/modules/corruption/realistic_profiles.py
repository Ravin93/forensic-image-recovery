# app/modules/corruption/realistic_profiles.py

import cv2
import numpy as np
import random


def generate_irregular_mask(shape, scale=0.3):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    num_blobs = random.randint(3, 8)

    for _ in range(num_blobs):
        x = random.randint(0, w)
        y = random.randint(0, h)
        radius = random.randint(10, int(min(h, w) * scale))

        cv2.circle(mask, (x, y), radius, 255, -1)

    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def apply_missing_region(image, level=30):
    mask = generate_irregular_mask(image.shape)

    corrupted = image.copy()
    corrupted[mask > 0] = 0

    return corrupted, mask


def apply_noise_region(image, level=30):
    mask = generate_irregular_mask(image.shape)

    noise = np.random.normal(0, level, image.shape).astype(np.uint8)

    corrupted = image.copy()
    corrupted[mask > 0] = noise[mask > 0]

    return corrupted, mask


def apply_blur_region(image, level=5):
    mask = generate_irregular_mask(image.shape)

    blurred = cv2.GaussianBlur(image, (level*2+1, level*2+1), 0)

    corrupted = image.copy()
    corrupted[mask > 0] = blurred[mask > 0]

    return corrupted, mask


def apply_jpeg_artifacts(image, strength=30):
    h, w = image.shape[:2]
    corrupted = image.copy()
    mask = np.zeros((h, w), dtype=np.uint8)

    block_size = 8

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            if random.random() < strength / 100:
                block = corrupted[i:i+block_size, j:j+block_size]

                if random.random() < 0.5:
                    block[:] = 0
                else:
                    block[:] = np.random.randint(0, 255)

                mask[i:i+block_size, j:j+block_size] = 255

    return corrupted, mask


def apply_mixed_corruption(image):
    funcs = [
        apply_missing_region,
        apply_noise_region,
        apply_blur_region,
        apply_jpeg_artifacts
    ]

    func = random.choice(funcs)
    return func(image)