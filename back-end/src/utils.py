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
