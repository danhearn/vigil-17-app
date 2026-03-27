import os
import asyncio
import threading
from pathlib import Path
from typing import Optional
import gi
from pythonosc.udp_client import SimpleUDPClient
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import numpy as np
import hailo
import cv2
import websockets
from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.hailo_app_python.apps.depth.depth_pipeline import GStreamerDepthApp

DEFAULT_NODE_SERVER_URL = "wss://stealth-composition.fly.dev/api/ws"

_osc_client           = SimpleUDPClient("127.0.0.1", 57120)
GLOBAL_GRAD_THRESHOLD = 50   # global gradient peak to fire a sample trigger

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
        self.prev_above_threshold = False

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
    string_to_print = f"Frame count: {count}\n"

    # Get the caps from the pad
    format, width, height = get_caps_from_pad(pad)
    frame = None

    # If use_frame flag is set to True
    if user_data.use_frame and format is not None and width is not None and height is not None:
        # Get video frame
        frame = get_numpy_from_buffer(buffer, format, width, height)
        print("Frame obtained")
        
        #cv2.imwrite(f"frames/frame_{count}.jpg", frame)
        # frame = 255 - frame
        # pil_img = Image.fromarray(frame).convert("L")
        # draw = ImageDraw.Draw(pil_img)
        #frame = np.array(pil_img)

    roi = hailo.get_roi_from_buffer(buffer)
    depth_mat = roi.get_objects_typed(hailo.HAILO_DEPTH_MASK)
    depth_mat = depth_mat[0]
    depth_mat = depth_mat.get_data()
    depth_mat = np.array(depth_mat).reshape((256, 320))    
    print(depth_mat[10][10])

    # if len(depth_mat) > 0:
    #     detection_average_depth = user_data.calculate_average_depth(depth_mat[0].get_data())
    # else:
    #     detection_average_depth = 0
    # string_to_print += (f"average depth: {detection_average_depth:.2f}\n")
    # print(string_to_print)
    depth_norm = cv2.normalize(depth_mat, None, 0, 255, cv2.NORM_MINMAX)
    depth_norm = depth_norm.astype(np.uint8)

    # Sample trigger: detect passing signal modulators via global gradient peak.
    # Rising edge fires /sample_trigger with lateral position (0=left, 1=right).
    sobel_x  = cv2.Sobel(depth_norm, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y  = cv2.Sobel(depth_norm, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(sobel_x, sobel_y)
    global_max_grad = float(grad_mag.max())
    above = global_max_grad >= GLOBAL_GRAD_THRESHOLD
    if above and not user_data.prev_above_threshold:
        h, w = grad_mag.shape
        _, _, _, max_loc = cv2.minMaxLoc(grad_mag)
        lateral_pos = float(max_loc[0]) / (w - 1)  # 0 = left, 1 = right
        _osc_client.send_message("/sample_trigger", [lateral_pos])
    user_data.prev_above_threshold = above

    # STREAM TO FRONT-END via background websocket client.
    stream_depth_frame(depth_norm)

    depth_colour = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
    output_path = f"frames/depth_colormap_{count}.jpg"
    cv2.imwrite(output_path, depth_colour)
    return Gst.PadProbeReturn.OK

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    env_file     = project_root / ".env"
    env_path_str = str(env_file)
    os.environ["HAILO_ENV_FILE"] = env_path_str

    user_data = user_app_callback_class()
    app = GStreamerDepthApp(app_callback, user_data)
    app.run()