"""
band_test.py — drive the SC patch with a synthetic contour, no camera or depth frames needed.

A bump (a "passer-by") walks left→right across the frame, producing the same
/wavetable and /bands messages as depth_stream.py.

Usage:
  python band_test.py            # 2 s per crossing
  python band_test.py 6          # slower crossing
"""

import sys
import time
import numpy as np
from pythonosc.udp_client import SimpleUDPClient

from src.utils import band_sliders, normalise_11

client = SimpleUDPClient("127.0.0.1", 57120)

height, width = 256, 320
n_bands       = 8
fps           = 24
crossing_secs = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0

x_arr = np.arange(width)
t0    = time.time()

try:
    while True:
        t = time.time() - t0

        # Bump position sweeps left→right, then the frame is empty for one crossing
        phase  = (t / crossing_secs) % 2.0
        centre = phase * width
        bump   = 100 * np.exp(-((x_arr - centre) ** 2) / (2 * 25 ** 2)) if phase < 1 else 0

        # Contour sits mid-frame with a little jitter; the bump lifts it (smaller y = higher)
        line = height / 2 - bump + np.random.normal(0, 2, width)

        sliders = band_sliders(line, height, n_bands)
        client.send_message("/bands", sliders.tolist())
        client.send_message("/wavetable", normalise_11(line).tolist())

        print(" ".join(f"{v:.2f}" for v in sliders), end="\r", flush=True)
        time.sleep(1 / fps)

except KeyboardInterrupt:
    print("\nStopped.")
