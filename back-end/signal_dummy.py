import numpy as np
import time
from pythonosc.udp_client import SimpleUDPClient

client = SimpleUDPClient("127.0.0.1", 57120)

while True:
    wt = np.random.uniform(-1, 1, 320)
    wt = wt / np.max(np.abs(wt))  # normalize
    client.send_message("/wavetable", wt.tolist())
    time.sleep(1/30)