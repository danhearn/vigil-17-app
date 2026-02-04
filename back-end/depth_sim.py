# Simulating depth matrix, sending to websocket
# Download the following folder, place in working dir: https://drive.google.com/drive/folders/1k7jPCid4u2xGjdC9YfnvGMTNsvHIv2qu?usp=sharing

import os
import time
import asyncio
import threading
from typing import Optional

import numpy as np
import cv2
import websockets

DEFAULT_NODE_SERVER_URL = "wss://stealth-composition.fly.dev/api/ws"


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
                print("Connection refused. Ensure Node.js server is running.")
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
        """Schedule frame send without blocking the main thread."""
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


# Example of depth mapping, will save local example jpg file
depth_norm = np.load("np_frames/frame_1.npy")
depth_norm = depth_norm.astype(np.uint8)
depth_colour = cv2.applyColorMap(depth_norm, cv2.COLORMAP_HSV)
output_path = "map_trial.jpg"
cv2.imwrite(output_path, depth_colour)

i = 0
fps = 30

# start websocket streamer
get_streamer()

try:
    while True:
        i = (i + 1) % 325
        depth_norm = np.load(f"np_frames/frame_{i+1}.npy")  # This is an np array, and can be sent to the websocket like you did previously with the pi
        depth_norm = depth_norm.astype(np.uint8)
        stream_depth_frame(depth_norm)
        print(f"frame {i+1} read")
        time.sleep(1 / fps)
except KeyboardInterrupt:
    pass

print("Script terminated")