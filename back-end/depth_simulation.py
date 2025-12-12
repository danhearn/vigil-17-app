# Script which simulates streaming depth matrices, corresponding to hailo dummy footage
# Download the following folder: https://drive.google.com/drive/folders/1k7jPCid4u2xGjdC9YfnvGMTNsvHIv2qu?usp=sharing
# Place in working directory, maintaining directory name

import time
import numpy as np
import cv2

# Example of depth mapping, will save local example jpg file
depth_norm = np.load("np_frames/frame_1.npy")
depth_norm = depth_norm.astype(np.uint8)
depth_colour = cv2.applyColorMap(depth_norm, cv2.COLORMAP_HSV)
output_path = "map_trial.jpg"
cv2.imwrite(output_path, depth_colour)

i = 0
fps = 30

try:
    while True:
        i = (i + 1) % 325
        depth_norm = np.load(f"np_frames/frame_{i+1}.npy") # This is an np array, and can be sent to the websocket like we did previously with the pi
        print(f'frame {i+1} read')
        time.sleep(1/fps)
except KeyboardInterrupt:
    pass

print('Script terminated')