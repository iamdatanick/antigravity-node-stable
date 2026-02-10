import { useHealth, countServices } from "../api/health";
import type { ServiceCheck } from "../api/health";
import { Server, Activity, CheckCircle, XCircle } from "lucide-react";

const LEVEL_LABELS: Record<string, string> = {
  L0: "Infrastructure",
  L1: "Orchestration",
  L2: "Services",
  L3: "Agent",
  L4: "Observability",
};

function ServiceRow({ check }: { check: ServiceCheck }) {
  const Icon = check.healthy ? CheckCircle : XCircle;
  const iconColor = check.healthy ? "text-[var(--color-green)]" : "text-[var(--color-red)]";

  return (
    <div className="flex items-center gap-3 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-secondary)] p-3 hover:border-[var(--color-accent)]/30 transition-colors">
      <Icon size={16} className={iconColor} />
      <div className="flex-1 min-w-0">
        <div className="text-sm font-medium text-[var(--color-text-primary)] truncate">
          {check.name}
        </div>
        <div className="text-xs text-[var(--color-text-muted)]">
          {check.healthy ? "healthy" : "unhealthy"}
        </div>
      </div>
      {check.error && (
        <span className="text-[10px] text-[var(--color-red)] truncate max-w-[120px]" title={check.error}>
          {check.error}
        </span>
      )}
    </div>
  );
}

export default function Services() {
  const { data, isLoading } = useHealth();

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Activity className="animate-spin text-[var(--color-accent)]" size={24} />
      </div>
    );
  }

  const svc = countServices(data);
  const levels = data?.levels || [];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-bold text-[var(--color-text-primary)] flex items-center gap-2">
          <Server size={20} />
          Services
        </h1>
        <p className="text-sm text-[var(--color-text-muted)]">
          {svc.total} containers — {svc.healthy} healthy, {svc.unhealthy} unhealthy
        </p>
      </div>

      {levels.length === 0 ? (
        <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-bg-secondary)] h-64 flex items-center justify-center">
          <div className="text-center">
            <Server size={48} className="mx-auto text-[var(--color-text-muted)] mb-3" />
            <p className="text-sm text-[var(--color-text-muted)]">No services detected</p>
          </div>
        </div>
      ) : (
        levels.map((level) => (
          <div key={level.level}>
            <h2 className="text-sm font-semibold text-[var(--color-text-secondary)] uppercase tracking-wider mb-3">
              {level.level} — {LEVEL_LABELS[level.level] || level.name}
              <span className="ml-2 text-[var(--color-text-muted)] font-normal normal-case">
                ({level.checks.length} services)
              </span>
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-2">
              {level.checks.map((check) => (
                <ServiceRow key={check.name} check={check} />
              ))}
            </div>
          </div>
        ))
      )}
    </div>
  );
}
