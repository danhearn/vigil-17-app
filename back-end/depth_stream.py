import os
import asyncio
import threading
from pathlib import Path
from typing import Optional
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import numpy as np
from scipy.signal.windows import tukey
from pythonosc.udp_client import SimpleUDPClient
import hailo
import cv2
import websockets
from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.hailo_app_python.apps.depth.depth_pipeline import GStreamerDepthApp

from src.utils import smooth_line, normalise_11

DEFAULT_NODE_SERVER_URL = "wss://stealth-composition.fly.dev/api/ws"

# Initialise UDP client
client = SimpleUDPClient("127.0.0.1", 57120)

# Defining parameters
fps            = 24
alpha          = 0.05       # background accumulation rate
tau            = 10         # gradient threshold for contour detection
temporal_beta  = 0.3        # temporal smoothing weight
gamma          = 0.5        # ghostly depth visualisation gamma
sine_freq      = 2          # sine wave cycles across frame width
sine_amp       = 30         # sine wave amplitude in pixels
weight         = 0.5        # sine/contour mixing coeffiecient
global_grad_threshold = 14

total_frames   = 1766
np_frames_dir  = Path("np_frames")
depth_dir      = Path("depth_frames")
depth_dir.mkdir(exist_ok=True)

# State parameters
background      = None
prev_frame_line = None
i               = -1

def normalize_ws_url(url: str) -> str:
    """
    Make sure we always end up with a ws:// or wss:// URI.
    Accept http(s):// values to avoid user error when copying REST URLs.
    """
    if url.startswith(("ws://", "wss://")):
        return url
    if url.startswith("https://"):
        return f"wss://{url[len('https://'):]}"
    if url.startswith("http://"):
        return f"ws://{url[len('http://'):]}"
    raise ValueError(
        f"NODE_SERVER_URL must start with ws:// or wss:// (got: {url})"
    )

NODE_SERVER_URL = normalize_ws_url(
    os.environ.get("NODE_SERVER_URL", DEFAULT_NODE_SERVER_URL)
)

try:
    STREAMING_INTERVAL = float(os.environ.get("STREAMING_INTERVAL", "0.033"))
except ValueError:
    STREAMING_INTERVAL = 0.033

try:
    STREAMING_QUEUE_SIZE = int(os.environ.get("STREAMING_QUEUE_SIZE", "5"))
except ValueError:
    STREAMING_QUEUE_SIZE = 5


class DepthStreamer:
    """Background asyncio client that keeps a single websocket alive."""

    def __init__(self, url: str):
        self.url = url
        self.loop = asyncio.new_event_loop()
        self.queue: Optional[asyncio.Queue[bytes]] = None
        self.queue_ready = threading.Event()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.queue = asyncio.Queue(maxsize=STREAMING_QUEUE_SIZE)
        self.queue_ready.set()
        self.loop.create_task(self._run())
        self.loop.run_forever()

    async def _run(self):
        while True:
            try:
                async with websockets.connect(self.url) as websocket:
                    print("Successfully connected. Streaming frames.")
                    await self._send_frames(websocket)
            except ConnectionRefusedError:
                print("Connection refused. Ensure Node.js server is running on port 3000.")
                await asyncio.sleep(3)
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed by the server. Attempting reconnect in 3 seconds...")
                await asyncio.sleep(3)
            except Exception as e:
                print(f"An unexpected error occurred in depth streamer: {e}")
                await asyncio.sleep(5)

    async def _send_frames(self, websocket):
        assert self.queue is not None
        while True:
            frame = await self.queue.get()
            try:
                await websocket.send(frame)
                await asyncio.sleep(STREAMING_INTERVAL)
            finally:
                self.queue.task_done()

    def enqueue(self, frame: bytes):
        """Schedule frame send without blocking the GStreamer thread."""
        if not self.queue_ready.is_set():
            return

        def _put():
            if self.queue is None:
                return
            if self.queue.full():
                print("Depth streamer queue full. Dropping frame.")
                return
            self.queue.put_nowait(frame)

        self.loop.call_soon_threadsafe(_put)


_STREAMER: Optional[DepthStreamer] = None
_STREAMER_LOCK = threading.Lock()


def get_streamer() -> DepthStreamer:
    global _STREAMER
    if _STREAMER is None:
        with _STREAMER_LOCK:
            if _STREAMER is None:
                _STREAMER = DepthStreamer(NODE_SERVER_URL)
    return _STREAMER


def stream_depth_frame(depth_frame: np.ndarray):
    """Convert the frame to bytes and hand it to the background streamer."""
    streaming_frame = np.ascontiguousarray(depth_frame).astype(np.uint8)
    get_streamer().enqueue(streaming_frame.tobytes())

# User-defined class to be used in the callback function: Inheritance from the app_callback_class
class user_app_callback_class(app_callback_class):

    def __init__(self):
        super().__init__()

    def calculate_average_depth(self, depth_mat):
        depth_values = np.array(depth_mat).flatten()  # Flatten the array and filter out outlier pixels
        try:
            m_depth_values = depth_values[depth_values <= np.percentile(depth_values, 95)]  # drop 5% of highest values (outliers)          
        except Exception as e:
            m_depth_values = np.array([])
        if len(m_depth_values) > 0:
            average_depth = np.mean(m_depth_values)  # Calculate the average depth of the pixels
        else:
            average_depth = 0  # Default value if no valid pixels are found
        return average_depth

# User-defined callback function: This is the callback function that will be called when data is available from the pipeline
def app_callback(pad, info, user_data):
    
    # Get the GstBuffer from the probe info
    buffer = info.get_buffer()
    if buffer is None: 
        return Gst.PadProbeReturn.OK
    
    # Using the user_data to count the number of frame
    user_data.increment()
    count = user_data.get_count()

    # Get the caps from the pad
    format, width, height = get_caps_from_pad(pad)
    frame = None

    # If use_frame flag is set to True
    if user_data.use_frame and format is not None and width is not None and height is not None:
        # Get video frame
        frame = get_numpy_from_buffer(buffer, format, width, height)
        print("Frame obtained")
        
    # HAILO depth matrix extraction pipeline 
    roi = hailo.get_roi_from_buffer(buffer)
    depth_mat = roi.get_objects_typed(hailo.HAILO_DEPTH_MASK)
    depth_mat = depth_mat[0]
    depth_mat = depth_mat.get_data()
    depth_mat = np.array(depth_mat).reshape((256, 320))    
    depth_norm = cv2.normalize(depth_mat, None, 0, 255, cv2.NORM_MINMAX)
    depth_norm = depth_norm.astype(np.uint8)

    # STREAM TO FRONT-END via background websocket client.
    stream_depth_frame(depth_norm)

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

    # Sample trigger on rising global gradient threshold, with peak modulating stereo position
    global_max_grad = float(grad_mag.max())
    above = global_max_grad >= global_grad_threshold
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

    # Implementing contour function f
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

    # Superposition of sine wave
    temporal_1d = temporal_line.flatten()
    centre_y    = height // 2
    x_arr       = np.arange(width)
    sine_wave   = centre_y + sine_amp * np.sin(
                      2 * np.pi * sine_freq * x_arr / width)
    # Applying tukey window to avoid clipping
    alpha = 0.2         # taper parameter 
    max_val = 0.5        # optional scaling

    weight = tukey(width, alpha=alpha) * max_val   

    # Superposition function (by addition)
    displaced = (1 - weight) * sine_wave + weight * temporal_1d

    # Normalising between [-1, 1]
    displaced_n = normalise_11(displaced)

    client.send_message("/wavetable", displaced_n.tolist())
    
    return Gst.PadProbeReturn.OK

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    env_file     = project_root / ".env"
    env_path_str = str(env_file)
    os.environ["HAILO_ENV_FILE"] = env_path_str

    user_data = user_app_callback_class()
    app = GStreamerDepthApp(app_callback, user_data)
    app.run()