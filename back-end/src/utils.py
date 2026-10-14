import cv2
import numpy as np

def smooth_line(y_positions, kernel_size=15):
    """1D Gaussian smoothing of y positions"""
    y_positions = np.array(y_positions, dtype=np.float32)
    kernel = cv2.getGaussianKernel(kernel_size, -1)
    smoothed = cv2.filter2D(y_positions, -1, kernel[:, 0])
    return smoothed.astype(np.int32)


def normalise_11(arr):
    """Normalise a 1D array to the range [-1, 1]."""
    a_min, a_max = arr.min(), arr.max()
    return 2 * (arr - a_min) / (a_max - a_min + 1e-8) - 1


def band_sliders(line, height, n_bands=8):
    """Split a contour line into vertical bands and return each band's mean
    height in [0, 1] (0 = bottom of frame, 1 = top)."""
    bands = np.array_split(np.asarray(line, dtype=np.float32), n_bands)
    means = np.array([b.mean() for b in bands])
    return 1.0 - np.clip(means / (height - 1), 0.0, 1.0)
