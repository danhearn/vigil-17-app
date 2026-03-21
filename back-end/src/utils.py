import cv2
import numpy as np

def smooth_line(y_positions, kernel_size=15):
    """1D Gaussian smoothing of y positions"""
    y_positions = np.array(y_positions, dtype=np.float32)
    kernel = cv2.getGaussianKernel(kernel_size, -1)
    smoothed = cv2.filter2D(y_positions, -1, kernel[:, 0])
    return smoothed.astype(np.int32)
