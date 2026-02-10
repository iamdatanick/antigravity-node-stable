import { useHealth, countServices } from "../api/health";
import HealthGrid from "../components/dashboard/HealthGrid";
import ServiceCard from "../components/dashboard/ServiceCard";
import { Activity, Server, Layers, Heart } from "lucide-react";

function StatCard({
  icon: Icon,
  label,
  value,
  accent,
}: {
  icon: typeof Activity;
  label: string;
  value: string | number;
  accent?: string;
}) {
  return (
    <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-bg-secondary)] p-4">
      <div className="flex items-center gap-2 mb-1">
        <Icon size={14} className={accent || "text-[var(--color-text-muted)]"} />
        <span className="text-xs text-[var(--color-text-muted)]">{label}</span>
      </div>
      <div className="text-2xl font-bold text-[var(--color-text-primary)]">{value}</div>
    </div>
  );
}

export default function Dashboard() {
  const { data, isLoading, isError } = useHealth();

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Activity className="animate-spin text-[var(--color-accent)]" size={24} />
        <span className="ml-3 text-[var(--color-text-secondary)]">Loading system health...</span>
      </div>
    );
  }

  if (isError || !data) {
    return (
      <div className="flex flex-col items-center justify-center h-64 gap-3">
        <div className="w-12 h-12 rounded-full bg-[var(--color-red-dim)] flex items-center justify-center">
          <Activity className="text-[var(--color-red)]" size={20} />
        </div>
        <p className="text-[var(--color-text-secondary)]">Cannot reach orchestrator</p>
        <p className="text-xs text-[var(--color-text-muted)]">
          Make sure the stack is running: docker compose up -d
        </p>
      </div>
    );
  }

  const svc = countServices(data);
  const allChecks = data.levels.flatMap((l) => l.checks);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-bold text-[var(--color-text-primary)]">System Dashboard</h1>
        <p className="text-sm text-[var(--color-text-muted)]">
          Antigravity Node v13.0 — Real-time system health
        </p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard icon={Heart} label="Status" value={data.status} accent="text-[var(--color-green)]" />
        <StatCard icon={Server} label="Services" value={`${svc.healthy}/${svc.total}`} accent="text-[var(--color-accent)]" />
        <StatCard icon={Layers} label="Layers" value={data.levels.length} accent="text-[var(--color-purple)]" />
        <StatCard icon={Activity} label="Unhealthy" value={svc.unhealthy} accent={svc.unhealthy > 0 ? "text-[var(--color-red)]" : "text-[var(--color-green)]"} />
      </div>

      <div>
        <h2 className="text-sm font-semibold text-[var(--color-text-secondary)] uppercase tracking-wider mb-3">
          Layer Health
        </h2>
        <HealthGrid data={data} />
      </div>

      <div>
        <h2 className="text-sm font-semibold text-[var(--color-text-secondary)] uppercase tracking-wider mb-3">
          All Services ({svc.total})
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-2">
          {allChecks.map((check) => (
            <ServiceCard key={check.name} check={check} />
          ))}
        </div>
      </div>
    </div>
  );
}
