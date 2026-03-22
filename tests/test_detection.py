import numpy as np

from app.modules.detection.fusion import fuse_detection_masks


def test_fuse_detection_masks_union():
    a = np.zeros((10, 10), dtype=np.uint8)
    b = np.zeros((10, 10), dtype=np.uint8)
    a[1:3, 1:3] = 255
    b[5:7, 5:7] = 255

    fused = fuse_detection_masks([a, b], strategy="union")
    assert fused.sum() > 0