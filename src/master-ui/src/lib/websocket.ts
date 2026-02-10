/** WebSocket with auto-reconnect and exponential backoff */

interface LogSocketOptions {
  onMessage: (data: string) => void;
  onStatusChange: (connected: boolean) => void;
  onReconnecting?: (attempt: number) => void;
  maxRetries?: number;
}

export function createLogSocket(opts: LogSocketOptions) {
  const { onMessage, onStatusChange, onReconnecting, maxRetries = 10 } = opts;
  let ws: WebSocket | null = null;
  let attempt = 0;
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  let destroyed = false;

  function connect() {
    if (destroyed) return;

    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(`${proto}//${location.host}/api/ws/logs`);

    ws.onopen = () => {
      attempt = 0;
      onStatusChange(true);
    };

    ws.onmessage = (e) => onMessage(e.data);

    ws.onerror = () => {
      // onclose will fire after onerror, handle reconnect there
    };

    ws.onclose = () => {
      onStatusChange(false);
      if (destroyed) return;
      if (attempt < maxRetries) {
        const delay = Math.min(1000 * Math.pow(2, attempt), 8000);
        attempt++;
        onReconnecting?.(attempt);
        reconnectTimer = setTimeout(connect, delay);
      }
    };
  }

  connect();

  return {
    destroy() {
      destroyed = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      ws?.close();
    },
    get connected() {
      return ws?.readyState === WebSocket.OPEN;
    },
  };
}
