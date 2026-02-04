# Simulating depth matrix for local use 
# Download the following folder: https://drive.google.com/drive/folders/1k7jPCid4u2xGjdC9YfnvGMTNsvHIv2qu?usp=sharing

import time
import numpy as np
import cv2
import os

# Example of depth mapping, will save local example jpg file
# depth_norm = np.load("np_frames/frame_1.npy")
# depth_norm = depth_norm.astype(np.uint8)
# depth_colour = cv2.applyColorMap(depth_norm, cv2.COLORMAP_HSV)
# output_path = "map_trial.jpg"
# cv2.imwrite(output_path, depth_colour)

i = 0
fps = 30

try:
    while True:
        i = (i + 1) % 325
        depth_norm = np.load(f"np_frames/frame_{i+1}.npy")  # This is an np array, and can be sent to the websocket like you did previously with the pi
        depth_norm = depth_norm.astype(np.uint8)
        print(f"frame {i+1} read")
        depth_colour = cv2.applyColorMap(depth_norm, cv2.COLORMAP_HSV)
        output_path = f'depth_frames/frame_{i+1}.jpg'
        cv2.imwrite(output_path, depth_colour)
        print(f'frame {i+1} saved as img')
        time.sleep(1 / fps)
        print
except KeyboardInterrupt:
    pass

print("Script terminated")