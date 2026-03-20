# Import third party libraries
import numpy as np
import cv2
import time

# Import util functions
from src.utils import smooth_line

fps = 120
edge_threshold = 10      # threshold for gradient magnitude to consider an edge
alpha = 0.05             # background running average
tau = 7             # minimum gradient to consider valid for line
temporal_beta = 0.2      # temporal smoothing factor (0-1)
i = 0
background = None
prev_frame_line = None   # for temporal smoothing

try:
    while True:
        i = (i + 1) % 325
        depth_norm = np.load(f"np_frames/frame_{i+1}.npy").astype(np.uint8)
        print(f"frame {i+1} read")

        # Adaptive background extraction, applying alpha value
        if background is None:
            background = depth_norm.astype(np.float32)

        cv2.accumulateWeighted(depth_norm, background, alpha)
        bg_uint8 = cv2.convertScaleAbs(background)
        fg = cv2.absdiff(depth_norm, bg_uint8)
        fg = cv2.medianBlur(fg, 3)  # remove small noise

        # Compute gradient magnitude using Sobel operators, smoothing the result
        sobel_x = cv2.Sobel(fg, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(fg, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(sobel_x, sobel_y)
        grad_mag = cv2.GaussianBlur(cv2.convertScaleAbs(grad_mag), (5,5), 0)

        # Initialising wavetable array, applying contour function
        height, width = grad_mag.shape
        y_positions = []
        prev_y = height // 2  # start line in middle if no previous value

        for x in range(width):
            column = grad_mag[:, x]
            # Applying max operator column-wise
            max_grad = column.max()
            # Applying contour logic
            if max_grad >= tau:
                y = np.argmax(column)
                prev_y = y
            y_positions.append(prev_y)

        # smooth within this frame
        y_smooth = smooth_line(y_positions, kernel_size=15)

        # Temporal smoothing across frames
        if prev_frame_line is None:
            temporal_line = y_smooth
        else:
            temporal_line = ((1 - temporal_beta) * prev_frame_line + temporal_beta * y_smooth).astype(np.int32)

        prev_frame_line = temporal_line  # store for next frame

        # Visualisation helper
        depth_gray = cv2.cvtColor(depth_norm, cv2.COLOR_GRAY2BGR)
        overlay = depth_gray.copy()

        # draw the temporally smoothed red line
        for x, y in enumerate(temporal_line):
            overlay[y, x] = [0, 0, 255]  # BGR red

        output_path = f'depth_frames/frame_{i+1}.jpg'
        cv2.imwrite(output_path, overlay)
        print(f'frame {i+1} saved as img')

        time.sleep(1 / fps)

except KeyboardInterrupt:
    pass