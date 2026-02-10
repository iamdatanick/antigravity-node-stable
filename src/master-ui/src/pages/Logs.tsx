import { useEffect, useRef, useState } from "react";
import { Terminal } from "@xterm/xterm";
import { FitAddon } from "@xterm/addon-fit";
import "@xterm/xterm/css/xterm.css";
import { createLogSocket } from "../lib/websocket";
import { ScrollText, Wifi, WifiOff, Trash2, Pause, Play } from "lucide-react";

export default function Logs() {
  const termRef = useRef<HTMLDivElement>(null);
  const terminalRef = useRef<Terminal | null>(null);
  const socketRef = useRef<ReturnType<typeof createLogSocket> | null>(null);
  const [connected, setConnected] = useState(false);
  const [paused, setPaused] = useState(false);
  const bufferRef = useRef<string[]>([]);

  useEffect(() => {
    if (!termRef.current) return;

    const term = new Terminal({
      theme: {
        background: "#0f172a",
        foreground: "#e2e8f0",
        cursor: "#00d4ff",
        selectionBackground: "rgba(0, 212, 255, 0.3)",
        black: "#1e293b",
        red: "#ef4444",
        green: "#22c55e",
        yellow: "#eab308",
        blue: "#3b82f6",
        magenta: "#8b5cf6",
        cyan: "#00d4ff",
        white: "#f1f5f9",
      },
      fontSize: 13,
      fontFamily: '"JetBrains Mono", "Fira Code", "Cascadia Code", monospace',
      cursorBlink: false,
      disableStdin: true,
      scrollback: 5000,
    });

    const fit = new FitAddon();
    term.loadAddon(fit);
    term.open(termRef.current);
    fit.fit();
    terminalRef.current = term;

    term.writeln("\x1b[36m[Antigravity Logs]\x1b[0m Connecting to WebSocket...");

    const socket = createLogSocket({
      onMessage: (data) => {
        if (paused) {
          bufferRef.current.push(data);
        } else {
          term.writeln(data);
        }
      },
      onStatusChange: (isConnected) => {
        setConnected(isConnected);
        if (isConnected) {
          term.writeln("\x1b[32m[Connected]\x1b[0m WebSocket established");
        }
      },
      onReconnecting: (attempt) => {
        term.writeln(`\x1b[33m[Reconnecting]\x1b[0m Attempt ${attempt}...`);
      },
    });
    socketRef.current = socket;

    const resizeObserver = new ResizeObserver(() => fit.fit());
    resizeObserver.observe(termRef.current);

    return () => {
      resizeObserver.disconnect();
      socket.destroy();
      term.dispose();
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!paused && bufferRef.current.length > 0 && terminalRef.current) {
      for (const line of bufferRef.current) {
        terminalRef.current.writeln(line);
      }
      bufferRef.current = [];
    }
  }, [paused]);

  const clearTerminal = () => terminalRef.current?.clear();
  const togglePause = () => setPaused((p) => !p);

  return (
    <div className="flex flex-col h-[calc(100vh-8rem)]">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h1 className="text-xl font-bold text-[var(--color-text-primary)] flex items-center gap-2">
            <ScrollText size={20} />
            Live Logs
          </h1>
          <p className="text-sm text-[var(--color-text-muted)]">
            Real-time log stream via WebSocket — auto-reconnects
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="flex items-center gap-1.5 text-xs">
            {connected ? (
              <>
                <Wifi size={12} className="text-[var(--color-green)]" />
                <span className="text-[var(--color-green)]">Connected</span>
              </>
            ) : (
              <>
                <WifiOff size={12} className="text-[var(--color-red)]" />
                <span className="text-[var(--color-red)]">Disconnected</span>
              </>
            )}
          </span>
          <button
            onClick={togglePause}
            className="flex items-center gap-1 text-xs px-2 py-1 rounded border border-[var(--color-border)] text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] hover:bg-[var(--color-bg-tertiary)] transition-colors"
          >
            {paused ? <Play size={12} /> : <Pause size={12} />}
            {paused ? "Resume" : "Pause"}
          </button>
          <button
            onClick={clearTerminal}
            className="flex items-center gap-1 text-xs px-2 py-1 rounded border border-[var(--color-border)] text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] hover:bg-[var(--color-bg-tertiary)] transition-colors"
          >
            <Trash2 size={12} />
            Clear
          </button>
        </div>
      </div>
      <div
        ref={termRef}
        className="flex-1 rounded-xl border border-[var(--color-border)] bg-[#0f172a] overflow-hidden"
      />
    </div>
  );
}
