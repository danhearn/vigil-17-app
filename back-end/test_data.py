import asyncio
import os
import websockets
import time
import numpy as np

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

WIDTH = 256
HEIGHT = 320
HEADER_BYTES = 4
BUFFER_SIZE = WIDTH * HEIGHT  # 81,920 pixels/payload bytes
MESSAGE_SIZE = HEADER_BYTES + BUFFER_SIZE
STREAMING_INTERVAL = 1/30 
streaming_data = bytearray(MESSAGE_SIZE)
frame_counter = 0

async def stream_data_to_server():
    """
    Connects to the Node.js WebSocket server and streams the 256x320 
    ArrayBuffer data at 30 FPS.
    """
    global frame_counter
    print(f"Attempting to connect to Node.js broker at {NODE_SERVER_URL}...")
    
    try:
        async with websockets.connect(NODE_SERVER_URL) as websocket:
            print("Successfully connected. Starting 30 FPS data push.")

            while True:
                frame = np.random.randint(0, 256, size=(HEIGHT, WIDTH), dtype=np.uint8)
                streaming_data = frame.tobytes()
                await websocket.send(streaming_data)
                await asyncio.sleep(STREAMING_INTERVAL)

    except ConnectionRefusedError:
        print(f"Connection refused. Ensure Node.js server is running on port 3000.")
    except websockets.exceptions.ConnectionClosed:
        print("Connection closed by the server. Attempting reconnect in 3 seconds...")
        await asyncio.sleep(3)
        await stream_data_to_server()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        await asyncio.sleep(5)
        await stream_data_to_server()

async def main():
    await stream_data_to_server()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStreamer client stopped manually.")