"""
audio_stream_web.py — Streams SuperCollider audio output to the relay server.

Prerequisites (one-time Pi setup):
  1. Load the ALSA loopback kernel module:
       sudo modprobe snd-aloop
     To persist across reboots:
       echo "snd-aloop" | sudo tee -a /etc/modules

  2. Find the loopback card number:
       aplay -l | grep Loopback
     Note the card index (e.g. "card 2: Loopback").

  3. Configure SuperCollider to output to the loopback device BEFORE s.boot:
       s.options.outDevice = "Loopback Audio (hw:Loopback,0)";
       s.boot;
     (Run ServerOptions.devices to list exact device names on your system.)

  4. Set ALSA_DEVICE env var if your loopback card index differs from the default:
       export ALSA_DEVICE="hw:Loopback,1"

Usage:
  python audio_stream_web.py
"""

import asyncio
import subprocess
import os

import websockets

DEFAULT_AUDIO_SERVER_URL = "wss://stealth-composition.fly.dev/api/audio-ws"
AUDIO_SERVER_URL = os.environ.get("AUDIO_SERVER_URL", DEFAULT_AUDIO_SERVER_URL)

# ffmpeg reads from device 1 of the loopback — the capture side of what SC plays to device 0.
ALSA_DEVICE = os.environ.get("ALSA_DEVICE", "hw:Loopback,1")

SAMPLE_RATE = 44100
CHANNELS = 2
# 4096 bytes = 1024 stereo frames ≈ 23 ms per chunk
CHUNK_BYTES = 4096


def _start_ffmpeg() -> subprocess.Popen:
    return subprocess.Popen(
        [
            "ffmpeg",
            "-f", "alsa",
            "-i", ALSA_DEVICE,
            "-ar", str(SAMPLE_RATE),
            "-ac", str(CHANNELS),
            "-f", "s16le",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )


async def _read_chunk(proc: subprocess.Popen) -> bytes:
    """Read one chunk from ffmpeg stdout without blocking the event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, proc.stdout.read, CHUNK_BYTES)


async def run():
    ffmpeg = _start_ffmpeg()
    print(f"ffmpeg capturing from {ALSA_DEVICE} at {SAMPLE_RATE}Hz {CHANNELS}ch")

    try:
        while True:
            try:
                async with websockets.connect(AUDIO_SERVER_URL) as ws:
                    print(f"Audio stream connected to {AUDIO_SERVER_URL}")
                    while True:
                        chunk = await _read_chunk(ffmpeg)
                        if not chunk:
                            print("ffmpeg stdout closed — restarting capture.")
                            ffmpeg.terminate()
                            ffmpeg = _start_ffmpeg()
                            break
                        await ws.send(chunk)
            except (ConnectionRefusedError, websockets.exceptions.ConnectionClosed) as e:
                print(f"Audio WS disconnected ({e}), retrying in 3 s…")
                await asyncio.sleep(3)
            except Exception as e:
                print(f"Unexpected audio stream error: {e}, retrying in 5 s…")
                await asyncio.sleep(5)
    finally:
        ffmpeg.terminate()


if __name__ == "__main__":
    asyncio.run(run())
