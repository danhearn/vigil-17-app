"use client"
import { useState, useEffect } from 'react';

const PREVIEW_VALUES = 1000;
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
  const [messages, setMessages] = useState<string>("Waiting for data...");

  useEffect(() => {
    if (!webSocket) return;

    const handleMessage = (event: MessageEvent) => {
      let messageContent: string;

      if (event.data instanceof ArrayBuffer) {
        const bytes = new Uint8Array(event.data);
        const preview = Array.from(bytes.slice(0, PREVIEW_VALUES));
        const suffix = bytes.length > PREVIEW_VALUES ? ` … (${bytes.length} total values)` : "";
        messageContent = `[${preview.join(", ")}]${suffix}`;
      } else if (typeof event.data === "string") {
        if (event.data === "connection established") return;
        messageContent = event.data;
      } else {
        messageContent = `Received unexpected non-string/non-ArrayBuffer data: ${typeof event.data}`;
      }
      
      setMessages(messageContent);
    };

    webSocket.addEventListener("message", handleMessage);
    return () => {
      webSocket?.removeEventListener("message", handleMessage);
    };
  }, []);

  if (!webSocket) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <p className="text-black dark:text-zinc-50">Initializing WebSocket...</p>
      </div>
    );
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-zinc-50 font-sans dark:bg-black">
      <main className="flex min-h-screen w-full max-w-3xl flex-col items-center justify-between py-32 px-16 bg-white dark:bg-black sm:items-start">
        <div className="flex flex-col items-center gap-6 text-center sm:items-start sm:text-left">
          <h1 className="max-w-xs text-3xl font-semibold leading-10 tracking-tight text-black dark:text-zinc-50">
            Stealth Composition
          </h1>
          <p className="max-w-md text-lg leading-8 text-zinc-600 dark:text-zinc-400">
            Messages: <span className="font-mono text-black dark:text-zinc-50">{messages}</span>
          </p>
        </div>
      </main>
    </div>
  );
}