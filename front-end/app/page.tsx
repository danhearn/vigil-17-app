"use client"
import { useState, useEffect, useRef } from 'react';

const FRAME_WIDTH = 320;
const FRAME_HEIGHT = 256;
const AUDIO_SAMPLE_RATE = 44100;
const AUDIO_CHANNELS = 2;
// Schedule each chunk 80 ms ahead to absorb jitter without audible delay.
const AUDIO_SCHEDULE_AHEAD_S = 0.08;

let webSocket: WebSocket | null = null;
let audioSocket: WebSocket | null = null;

if (typeof window !== "undefined") {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";

  const establishConnection = (path: string, onClose: () => void) => {
    const ws = new WebSocket(`${protocol}//${window.location.host}${path}`);
    ws.binaryType = 'arraybuffer';
    ws.onclose = () => {
      console.log(`${path} closed. Reconnecting…`);
      setTimeout(onClose, 3000);
    };
    ws.onerror = (e) => console.error(`${path} error:`, e);
    return ws;
  };

  const connectVideo = () => {
    webSocket = establishConnection('/api/ws', connectVideo);
  };
  const connectAudio = () => {
    audioSocket = establishConnection('/api/audio-ws', connectAudio);
  };

  connectVideo();
  connectAudio();
}

export default function Home() {
  const [frame, setFrame] = useState<Uint8Array | null>(null);
  const [audioStarted, setAudioStarted] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const nextAudioTimeRef = useRef(0);

  // Video WebSocket
  useEffect(() => {
    if (!webSocket) return;

    const handleMessage = (event: MessageEvent) => {
      if (event.data instanceof ArrayBuffer) {
        setFrame(new Uint8Array(event.data));
        return;
      }
      if (typeof event.data === "string") {
        if (event.data === "connection established") return;
        console.debug("WebSocket message:", event.data);
        return;
      }
      console.warn("Received unexpected payload:", typeof event.data);
    };

    webSocket.addEventListener("message", handleMessage);
    return () => webSocket?.removeEventListener("message", handleMessage);
  }, []);

  // Audio WebSocket — plays back raw s16le stereo PCM chunks via Web Audio API.
  // AudioContext is created on first user interaction to satisfy browser autoplay policy.
  useEffect(() => {
    if (!audioSocket) return;

    const handleAudioMessage = (event: MessageEvent) => {
      if (!(event.data instanceof ArrayBuffer)) return;

      // Lazily create AudioContext on first audio chunk after user has interacted.
      if (!audioCtxRef.current) return;
      const ctx = audioCtxRef.current;

      const int16 = new Int16Array(event.data);
      const frameCount = int16.length / AUDIO_CHANNELS;
      const buffer = ctx.createBuffer(AUDIO_CHANNELS, frameCount, AUDIO_SAMPLE_RATE);

      // Deinterleave interleaved stereo s16le → float32 per channel
      const left = buffer.getChannelData(0);
      const right = buffer.getChannelData(1);
      for (let i = 0; i < frameCount; i++) {
        left[i]  = int16[i * 2]     / 32768;
        right[i] = int16[i * 2 + 1] / 32768;
      }

      const source = ctx.createBufferSource();
      source.buffer = buffer;
      source.connect(ctx.destination);

      // Schedule contiguously; clamp to now + schedule-ahead to recover from gaps.
      const startTime = Math.max(
        ctx.currentTime + AUDIO_SCHEDULE_AHEAD_S,
        nextAudioTimeRef.current
      );
      source.start(startTime);
      nextAudioTimeRef.current = startTime + buffer.duration;
    };

    audioSocket.addEventListener("message", handleAudioMessage);
    return () => audioSocket?.removeEventListener("message", handleAudioMessage);
  }, []);

  // Start AudioContext on first user interaction (browser autoplay policy).
  const startAudio = () => {
    if (audioCtxRef.current) return;
    audioCtxRef.current = new AudioContext({ sampleRate: AUDIO_SAMPLE_RATE });
    nextAudioTimeRef.current = audioCtxRef.current.currentTime;
    setAudioStarted(true);
  };

  useEffect(() => {
    if (!frame || !canvasRef.current) return;
    if (frame.length !== FRAME_WIDTH * FRAME_HEIGHT) {
      console.warn(`Unexpected frame size: ${frame.length} bytes`);
      return;
    }

    const ctx = canvasRef.current.getContext("2d");
    if (!ctx) return;
    ctx.imageSmoothingEnabled = false;

    // Expand the single-channel buffer into RGBA for putImageData.
    const rgba = new Uint8ClampedArray(frame.length * 4);
    for (let i = 0, j = 0; i < frame.length; i++, j += 4) {
      const value = frame[i];
      rgba[j] = value;
      rgba[j + 1] = value;
      rgba[j + 2] = value;
      rgba[j + 3] = 255;
    }

    const imageData = new ImageData(rgba, FRAME_WIDTH, FRAME_HEIGHT);
    ctx.putImageData(imageData, 0, 0);
  }, [frame]);

  if (!webSocket) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <p className="text-black dark:text-zinc-50">Initializing WebSocket...</p>
      </div>
    );
  }

  return (
    <div
      className="relative flex min-h-screen w-screen items-center justify-center bg-zinc-50 font-sans dark:bg-black"
      onClick={startAudio}
    >
      {!frame && (
        <p className="absolute top-6 left-6 text-lg leading-8 text-zinc-600 dark:text-zinc-400 font-mono">
          Waiting for data...
        </p>
      )}
      {!audioStarted && (
        <p className="absolute bottom-6 left-6 text-sm text-zinc-500 dark:text-zinc-500 font-mono pointer-events-none">
          Click to enable audio
        </p>
      )}
      <canvas
        ref={canvasRef}
        width={FRAME_WIDTH}
        height={FRAME_HEIGHT}
        className="h-screen w-screen border-0 bg-black"
        style={{ imageRendering: "pixelated" }}
      />
    </div>
  );
}