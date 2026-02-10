import type { ServiceCheck } from "../../api/health";
import { CheckCircle, XCircle } from "lucide-react";

export default function ServiceCard({ check }: { check: ServiceCheck }) {
  const Icon = check.healthy ? CheckCircle : XCircle;
  const iconColor = check.healthy
    ? "text-[var(--color-green)]"
    : "text-[var(--color-red)]";
  const statusText = check.healthy ? "healthy" : "unhealthy";

  return (
    <div className="flex items-center gap-3 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-secondary)] p-3 hover:border-[var(--color-accent)]/30 transition-colors">
      <Icon size={16} className={iconColor} />
      <div className="flex-1 min-w-0">
        <div className="text-sm font-medium text-[var(--color-text-primary)] truncate">
          {check.name}
        </div>
        <div className="text-xs text-[var(--color-text-muted)]">{statusText}</div>
      </div>
      {check.error && (
        <span className="text-[10px] text-[var(--color-red)] truncate max-w-[120px]" title={check.error}>
          {check.error}
        </span>
      )}
    </div>
  );
}
