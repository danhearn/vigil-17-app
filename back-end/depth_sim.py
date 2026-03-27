import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal.windows import tukey
from pythonosc.udp_client import SimpleUDPClient

client = SimpleUDPClient("127.0.0.1", 57120)

# ── Parameters ────────────────────────────────────────────────────────────────
fps                    = 24
alpha                  = 0.05   # background accumulation rate
tau                    = 10     # gradient threshold for contour detection
GLOBAL_GRAD_THRESHOLD  = 50     # global gradient peak to fire a sample trigger
temporal_beta  = 0.3        # temporal smoothing weight
gamma          = 0.5        # ghostly depth visualisation gamma
sine_freq      = 2          # sine wave cycles across frame width
sine_amp       = 30         # sine wave amplitude in pixels
weight         = 0.5        # 0=pure sine, 1=pure depth contour

total_frames   = 1766
np_frames_dir  = Path("np_frames")
depth_dir      = Path("depth_frames")
depth_dir.mkdir(exist_ok=True)

# ── State ─────────────────────────────────────────────────────────────────────
background           = None
prev_frame_line      = None
i                    = -1
prev_above_threshold = False

# ── Helpers ───────────────────────────────────────────────────────────────────
def smooth_line(y_positions, kernel_size=15):
    """1D moving average smooth over y_positions list."""
    arr    = np.array(y_positions, dtype=np.float32)
    kernel = np.ones(kernel_size) / kernel_size
    padded = np.pad(arr, kernel_size // 2, mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[:len(arr)].astype(np.int32)

def normalise_11(arr):
    """Normalise a 1D array to the range [-1, 1]."""
    a_min, a_max = arr.min(), arr.max()
    return 2 * (arr - a_min) / (a_max - a_min + 1e-8) - 1

# ── Main loop ─────────────────────────────────────────────────────────────────
try:
    while True:
        i = (i + 1) % total_frames

        # Load raw float depth and apply ghostly normalisation
        depth  = np.load(np_frames_dir / f"frame_{i+1:06d}.npy")
        d_min, d_max = depth.min(), depth.max()
        normalised   = (depth - d_min) / (d_max - d_min + 1e-8)
        inverted     = 1.0 - normalised
        ghostly      = np.power(inverted, gamma)
        depth_norm   = (ghostly * 255).astype(np.uint8)
        print(f"frame {i+1} read")

        # Adaptive background extraction
        if background is None:
            background = depth_norm.astype(np.float32)

        cv2.accumulateWeighted(depth_norm, background, alpha)
        bg_uint8 = cv2.convertScaleAbs(background)
        fg       = cv2.absdiff(depth_norm, bg_uint8)
        fg       = cv2.medianBlur(fg, 3)

        # Gradient magnitude via Sobel
        sobel_x  = cv2.Sobel(fg, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y  = cv2.Sobel(fg, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(sobel_x, sobel_y)
        grad_mag = cv2.GaussianBlur(cv2.convertScaleAbs(grad_mag), (5, 5), 0)

        # Sample trigger: fire on the rising edge of the global gradient threshold.
        # A passing signal modulator (pedestrian / vehicle) produces a sharp peak;
        # the column of that peak maps to stereo pan position in SuperCollider.
        global_max_grad = float(grad_mag.max())
        above = global_max_grad >= GLOBAL_GRAD_THRESHOLD
        if above and not prev_above_threshold:
            max_row, max_col = np.unravel_index(grad_mag.argmax(), grad_mag.shape)
            lateral_pos = float(max_col) / (width - 1)  # 0 = left, 1 = right
            client.send_message("/sample_trigger", [lateral_pos])
            print(f"  /sample_trigger  lateral={lateral_pos:.2f}  grad={global_max_grad:.1f}")
        prev_above_threshold = above

        # Contour extraction — column-wise argmax above tau
        height, width = grad_mag.shape
        y_positions   = []
        prev_y        = height // 2

        for x in range(width):
            column   = grad_mag[:, x]
            max_grad = column.max()
            if max_grad >= tau:
                prev_y = np.argmax(column)
            y_positions.append(prev_y)

        # Spatial smoothing
        y_smooth = smooth_line(y_positions, kernel_size=15)

        # Temporal smoothing
        if prev_frame_line is None:
            temporal_line = y_smooth
        else:
            temporal_line = (
                (1 - temporal_beta) * prev_frame_line +
                temporal_beta * y_smooth
            ).astype(np.int32)

        prev_frame_line = temporal_line.flatten()   # keep 1D

        # ── Depth frame visualisation (red contour line) ──────────────────────
        depth_gray = cv2.cvtColor(depth_norm, cv2.COLOR_GRAY2BGR)
        overlay    = depth_gray.copy()

        for x, y in enumerate(temporal_line.flatten()):
            y_clipped = int(np.clip(y, 0, height - 1))
            overlay[y_clipped, x] = [0, 0, 255]   # BGR red

        cv2.imwrite(str(depth_dir / f"frame_{i+1:06d}.jpg"), overlay)
        print(f"frame {i+1} saved as img")

# ── Sine displacement plot ────────────────────────────────────────────
        temporal_1d = temporal_line.flatten()
        centre_y    = height // 2
        x_arr       = np.arange(width)
        sine_wave   = centre_y + sine_amp * np.sin(
                          2 * np.pi * sine_freq * x_arr / width)
        # --- Tukey window as weight ---
        alpha = 0.2         # taper parameter (expose this)
        max_val = 0.5        # optional scaling

        weight = tukey(width, alpha=alpha) * max_val   # shape: (width,)

        # --- blend ---
        displaced = (1 - weight) * sine_wave + weight * temporal_1d

        print(sine_wave[0])

        sine_n     = normalise_11(sine_wave)
        temporal_n = normalise_11(temporal_1d)
        displaced_n = normalise_11(displaced)

        client.send_message("/wavetable", displaced_n.tolist())

        time.sleep(0.5)

except KeyboardInterrupt:
    print(f"\nStopped at frame {i+1}. Output saved to {depth_dir}/")