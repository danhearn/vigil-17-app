#!/usr/bin/env python3
"""
run.py — launch the backend processes for the appropriate mode.

Usage:
  python run.py sim        # depth_sim.py only  (OSC → SuperCollider, no websocket)
  python run.py sim-web    # depth_sim_web.py + audio_stream_web.py
  python run.py live       # depth_stream_web.py + audio_stream_web.py  (Pi + Hailo)
"""

import sys
import signal
import subprocess
import threading
from pathlib import Path

HERE = Path(__file__).parent

MODES: dict[str, list[tuple[str, list[str]]]] = {
    "sim": [
        ("depth", [sys.executable, str(HERE / "depth_sim.py")]),
    ],
    "sim-web": [
        ("depth", [sys.executable, str(HERE / "depth_sim_web.py")]),
        ("audio", [sys.executable, str(HERE / "audio_stream_web.py")]),
    ],
    "live": [
        ("depth", [sys.executable, str(HERE / "depth_stream_web.py")]),
        ("audio", [sys.executable, str(HERE / "audio_stream_web.py")]),
    ],
}


def _stream(proc: subprocess.Popen, tag: str) -> None:
    """Forward a process's stdout+stderr to our stdout with a label prefix."""
    for line in proc.stdout:  # type: ignore[union-attr]
        print(f"[{tag}] {line}", end="", flush=True)


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in MODES:
        print(f"usage: python run.py <mode>\nmodes: {', '.join(MODES)}")
        sys.exit(1)

    mode = sys.argv[1]
    specs = MODES[mode]
    procs: list[subprocess.Popen] = []

    for tag, cmd in specs:
        p = subprocess.Popen(
            cmd,
            cwd=HERE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        threading.Thread(target=_stream, args=(p, tag), daemon=True).start()
        procs.append(p)
        print(f"[run] started {tag}  pid={p.pid}  {Path(cmd[1]).name}")

    def _shutdown(sig, frame) -> None:
        print("\n[run] shutting down…")
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Block until any child exits unexpectedly, then tear down the rest.
    done = threading.Event()
    def _watch(p: subprocess.Popen, tag: str) -> None:
        p.wait()
        print(f"[run] {tag} exited (code {p.returncode})")
        done.set()

    for p, (tag, _) in zip(procs, specs):
        threading.Thread(target=_watch, args=(p, tag), daemon=True).start()

    done.wait()
    _shutdown(None, None)


if __name__ == "__main__":
    main()
