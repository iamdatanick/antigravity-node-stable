import { useState, useEffect, useRef } from "react";
import { Activity, Cpu, DollarSign, Terminal, ChevronDown, Wifi, WifiOff } from "lucide-react";
import { useHealth, countServices } from "../../api/health";
import { useModels } from "../../api/models";
import { useBudget } from "../../api/budget";
import { createLogSocket } from "../../lib/websocket";

interface AmbientHUDProps {
  activeModel: string;
  onModelChange: (model: string) => void;
  isStreaming: boolean;
}

/**
 * Translucent ambient HUD overlays at viewport edges.
 * Top-left: model selector + health. Top-right: budget. Bottom-left: live logs.
 */
export default function AmbientHUD({ activeModel, onModelChange, isStreaming }: AmbientHUDProps) {
  const { data: healthData } = useHealth();
  const { data: modelsData } = useModels();
  const { data: budgetData } = useBudget();
  const [logs, setLogs] = useState<string[]>([]);
  const [logConnected, setLogConnected] = useState(false);
  const [showLogs, setShowLogs] = useState(false);
  const logRef = useRef<HTMLDivElement>(null);

  const models = modelsData?.data ?? [];
  const { total, healthy } = countServices(healthData);
  const healthPct = total > 0 ? Math.round((healthy / total) * 100) : 0;

  // WebSocket log stream
  useEffect(() => {
    const socket = createLogSocket({
      onMessage: (data) => {
        // Strip ANSI codes for clean display
        const clean = data.replace(/\x1b\[[0-9;]*m/g, "").trim();
        if (clean) {
          setLogs((prev) => [...prev.slice(-30), clean]);
        }
      },
      onStatusChange: setLogConnected,
      maxRetries: 5,
    });
    return () => socket.destroy();
  }, []);

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logs]);

  const budgetPct = budgetData ? Math.round((budgetData.spent_today / budgetData.daily_budget) * 100) : 0;
  const budgetRemaining = budgetData ? budgetData.remaining.toFixed(2) : "—";

  return (
    <>
      {/* Top-left: Model + Health */}
      <div className="fixed top-4 left-4 flex flex-col gap-2" style={{ zIndex: 20 }}>
        {/* Model selector */}
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-[rgba(15,23,42,0.7)] border border-[rgba(255,255,255,0.06)] backdrop-blur-md">
          <Cpu size={11} className="text-[var(--color-accent)] opacity-60" />
          <div className="relative">
            <select
              value={activeModel}
              onChange={(e) => onModelChange(e.target.value)}
              disabled={isStreaming}
              className="appearance-none bg-transparent text-[10px] font-mono text-[var(--color-text-secondary)] pr-5 focus:outline-none cursor-pointer disabled:opacity-40"
            >
              {models.length > 0 ? (
                models.map((m) => (
                  <option key={m.id} value={m.id} className="bg-[#0f172a] text-[var(--color-text-primary)]">
                    {m.id}
                  </option>
                ))
              ) : (
                <>
                  <option value="gpt-4o" className="bg-[#0f172a]">GPT-4o</option>
                  <option value="gpt-4o-mini" className="bg-[#0f172a]">GPT-4o Mini</option>
                  <option value="claude-sonnet-4-20250514" className="bg-[#0f172a]">Claude Sonnet 4</option>
                  <option value="claude-haiku-4-5-20251001" className="bg-[#0f172a]">Claude Haiku 4.5</option>
                </>
              )}
            </select>
            <ChevronDown size={9} className="absolute right-0 top-1/2 -translate-y-1/2 text-[var(--color-text-muted)] opacity-40 pointer-events-none" />
          </div>
        </div>

        {/* Health indicator */}
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-[rgba(15,23,42,0.7)] border border-[rgba(255,255,255,0.06)] backdrop-blur-md">
          <Activity size={11} className={healthPct === 100 ? "text-[var(--color-green)] opacity-70" : "text-[var(--color-yellow)] opacity-70"} />
          <span className="text-[10px] font-mono text-[var(--color-text-muted)]">
            {healthy}/{total}
          </span>
          <div className="w-12 h-1 rounded-full bg-[rgba(255,255,255,0.06)] overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-1000"
              style={{
                width: `${healthPct}%`,
                background: healthPct === 100 ? "var(--color-green)" : healthPct > 50 ? "var(--color-yellow)" : "var(--color-red)",
              }}
            />
          </div>
        </div>
      </div>

      {/* Top-right: Budget */}
      <div className="fixed top-4 right-4" style={{ zIndex: 20 }}>
        <div className="flex items-center gap-2.5 px-3 py-1.5 rounded-xl bg-[rgba(15,23,42,0.7)] border border-[rgba(255,255,255,0.06)] backdrop-blur-md">
          <DollarSign size={11} className="text-[var(--color-green)] opacity-60" />
          <div className="flex flex-col">
            <span className="text-[10px] font-mono text-[var(--color-text-secondary)]">
              ${budgetRemaining}
            </span>
            <div className="w-16 h-0.5 rounded-full bg-[rgba(255,255,255,0.06)] overflow-hidden mt-0.5">
              <div
                className="h-full rounded-full transition-all duration-1000"
                style={{
                  width: `${100 - budgetPct}%`,
                  background: budgetPct < 60 ? "var(--color-green)" : budgetPct < 85 ? "var(--color-yellow)" : "var(--color-red)",
                }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Bottom-left: Log tail (collapsible) */}
      <div className="fixed bottom-20 left-4" style={{ zIndex: 20 }}>
        <button
          onClick={() => setShowLogs((v) => !v)}
          className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-[rgba(15,23,42,0.7)] border border-[rgba(255,255,255,0.06)] backdrop-blur-md text-[9px] font-mono text-[var(--color-text-muted)] hover:text-[var(--color-text-secondary)] transition-colors"
        >
          <Terminal size={10} className="opacity-50" />
          {logConnected ? <Wifi size={8} className="text-[var(--color-green)] opacity-50" /> : <WifiOff size={8} className="text-[var(--color-red)] opacity-50" />}
          <span className="opacity-50">{showLogs ? "hide" : "logs"}</span>
        </button>

        {showLogs && (
          <div
            ref={logRef}
            className="mt-2 w-80 h-40 overflow-y-auto rounded-xl bg-[rgba(15,23,42,0.85)] border border-[rgba(255,255,255,0.06)] backdrop-blur-md p-2.5 space-y-0.5"
          >
            {logs.length === 0 ? (
              <span className="text-[9px] font-mono text-[var(--color-text-muted)] opacity-30">Waiting for logs...</span>
            ) : (
              logs.map((line, i) => (
                <div key={i} className="text-[9px] font-mono text-[var(--color-text-muted)] opacity-60 leading-tight break-all">
                  {line}
                </div>
              ))
            )}
          </div>
        )}
      </div>
    </>
  );
}
