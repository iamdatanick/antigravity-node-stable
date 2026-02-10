import { useHealth, countServices } from "../../api/health";
import { Activity, Wifi, WifiOff } from "lucide-react";

function StatusDot({ status }: { status: string }) {
  const color =
    status === "healthy"
      ? "bg-[var(--color-green)]"
      : status === "degraded"
        ? "bg-[var(--color-yellow)]"
        : "bg-[var(--color-red)]";
  const glow =
    status === "healthy"
      ? "shadow-[0_0_6px_var(--color-green)]"
      : status === "degraded"
        ? "shadow-[0_0_6px_var(--color-yellow)]"
        : "";
  return <span className={`inline-block w-2.5 h-2.5 rounded-full ${color} ${glow}`} />;
}

export default function TopBar() {
  const { data, isError, isFetching } = useHealth();
  const svc = countServices(data);

  return (
    <header className="h-14 flex items-center justify-between px-6 border-b border-[var(--color-border)] bg-[var(--color-bg-secondary)]">
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2">
          {data ? (
            <>
              <StatusDot status={data.status} />
              <span className="text-sm text-[var(--color-text-secondary)]">
                System {data.status}
              </span>
            </>
          ) : isError ? (
            <>
              <WifiOff size={14} className="text-[var(--color-red)]" />
              <span className="text-sm text-[var(--color-red)]">Disconnected</span>
            </>
          ) : (
            <>
              <Activity size={14} className="text-[var(--color-text-muted)] animate-pulse" />
              <span className="text-sm text-[var(--color-text-muted)]">Connecting...</span>
            </>
          )}
        </div>
        {isFetching && (
          <span className="text-[10px] text-[var(--color-text-muted)]">polling...</span>
        )}
      </div>
      <div className="flex items-center gap-4">
        <span className="text-xs px-2 py-0.5 rounded bg-[var(--color-accent-dim)] text-[var(--color-accent)] font-mono">
          v13.0
        </span>
        <div className="flex items-center gap-1.5">
          {svc.total > 0 ? (
            <Wifi size={14} className="text-[var(--color-green)]" />
          ) : (
            <WifiOff size={14} className="text-[var(--color-text-muted)]" />
          )}
          <span className="text-xs text-[var(--color-text-muted)]">
            {svc.healthy}/{svc.total} services
          </span>
        </div>
      </div>
    </header>
  );
}
