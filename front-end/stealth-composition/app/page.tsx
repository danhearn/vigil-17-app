"use client"
import { useState, useEffect, useRef } from 'react';

const FRAME_WIDTH = 320;
const FRAME_HEIGHT = 256;
let webSocket: WebSocket | null = null;

if (typeof window !== "undefined") {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/api/ws`;

  const establishConnection = () => {
    const newWs = new WebSocket(wsUrl);
    
    newWs.binaryType = 'arraybuffer';
    
    newWs.onclose = () => {
      console.log('WebSocket closed. Attempting reconnect...');
      setTimeout(() => {
        webSocket = establishConnection();
      }, 3000);
    };

    newWs.onerror = (error) => {
      console.error('WebSocket Error:', error);
    };

    return newWs;
  };
  
  webSocket = establishConnection();
}

export default function Home() {
  const [frame, setFrame] = useState<Uint8Array | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

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
    return () => {
      webSocket?.removeEventListener("message", handleMessage);
    };
  }, []);

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
    <div className="relative flex min-h-screen w-screen items-center justify-center bg-zinc-50 font-sans dark:bg-black">
      {!frame && (
        <p className="absolute top-6 left-6 text-lg leading-8 text-zinc-600 dark:text-zinc-400 font-mono">
          Waiting for data...
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