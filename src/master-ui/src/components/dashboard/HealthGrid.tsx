import type { HealthData } from "../../api/health";

const LEVEL_LABELS: Record<string, string> = {
  L0: "Infrastructure",
  L1: "Orchestration",
  L2: "Services",
  L3: "Agent",
  L4: "Observability",
};

const LEVEL_COLORS: Record<string, string> = {
  L0: "var(--color-accent)",
  L1: "var(--color-purple)",
  L2: "var(--color-green)",
  L3: "var(--color-yellow)",
  L4: "var(--color-red)",
};

function LevelBadge({ level }: { level: string }) {
  const bg = LEVEL_COLORS[level] || "var(--color-text-muted)";
  return (
    <span
      className="text-xs font-mono px-1.5 py-0.5 rounded text-white"
      style={{ backgroundColor: bg, opacity: 0.85 }}
    >
      {level}
    </span>
  );
}

export default function HealthGrid({ data }: { data: HealthData }) {
  const levels = data.levels || [];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {levels.map((level) => {
        const allHealthy = level.checks.every((c) => c.healthy);
        const someHealthy = level.checks.some((c) => c.healthy);
        const status = allHealthy ? "healthy" : someHealthy ? "degraded" : "unhealthy";
        const statusCls =
          status === "healthy"
            ? "bg-[var(--color-green-dim)] text-[var(--color-green)] border-[var(--color-green)]/20"
            : status === "degraded"
              ? "bg-[var(--color-yellow-dim)] text-[var(--color-yellow)] border-[var(--color-yellow)]/20"
              : "bg-[var(--color-red-dim)] text-[var(--color-red)] border-[var(--color-red)]/20";

        return (
          <div
            key={level.level}
            className="rounded-xl border border-[var(--color-border)] bg-[var(--color-bg-secondary)] p-4 hover:border-[var(--color-accent)]/30 transition-colors"
          >
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <LevelBadge level={level.level} />
                <span className="text-sm font-medium text-[var(--color-text-primary)]">
                  {LEVEL_LABELS[level.level] || level.name}
                </span>
              </div>
              <span className={`text-[10px] font-semibold uppercase px-2 py-0.5 rounded-full border ${statusCls}`}>
                {status}
              </span>
            </div>
            <div className="flex flex-wrap gap-1.5">
              {level.checks.map((check) => {
                const dot = check.healthy ? "bg-[var(--color-green)]" : "bg-[var(--color-red)]";
                return (
                  <span
                    key={check.name}
                    className="flex items-center gap-1 text-[11px] text-[var(--color-text-secondary)] bg-[var(--color-bg-primary)] px-2 py-1 rounded-md"
                    title={check.error || ""}
                  >
                    <span className={`w-1.5 h-1.5 rounded-full ${dot}`} />
                    {check.name}
                  </span>
                );
              })}
            </div>
          </div>
        );
      })}
    </div>
  );
}
