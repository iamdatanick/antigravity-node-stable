import { useBudget } from "../api/budget";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Cell,
} from "recharts";
import { DollarSign, Activity, TrendingDown, TrendingUp } from "lucide-react";

function StatCard({
  label,
  value,
  icon: Icon,
  color,
}: {
  label: string;
  value: string;
  icon: typeof DollarSign;
  color: string;
}) {
  return (
    <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-bg-secondary)] p-4">
      <div className="flex items-center gap-2 mb-1">
        <Icon size={14} className={color} />
        <span className="text-xs text-[var(--color-text-muted)]">{label}</span>
      </div>
      <div className="text-2xl font-bold text-[var(--color-text-primary)]">{value}</div>
    </div>
  );
}

export default function Budget() {
  const { data, isLoading, isError } = useBudget();

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Activity className="animate-spin text-[var(--color-accent)]" size={24} />
      </div>
    );
  }

  if (isError || !data) {
    return (
      <div className="space-y-4">
        <div>
          <h1 className="text-xl font-bold text-[var(--color-text-primary)] flex items-center gap-2">
            <DollarSign size={20} />
            Budget Tracker
          </h1>
        </div>
        <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-bg-secondary)] h-64 flex items-center justify-center">
          <p className="text-sm text-[var(--color-text-muted)]">Cannot load budget data</p>
        </div>
      </div>
    );
  }

  const pctUsed = data.daily_budget > 0 ? (data.spent_today / data.daily_budget) * 100 : 0;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-bold text-[var(--color-text-primary)] flex items-center gap-2">
          <DollarSign size={20} />
          Budget Tracker
        </h1>
        <p className="text-sm text-[var(--color-text-muted)]">
          LLM spend monitoring — daily budget enforcement ({data.currency})
        </p>
      </div>

      <div className="grid grid-cols-3 gap-4">
        <StatCard
          label="Daily Budget"
          value={`$${data.daily_budget.toFixed(2)}`}
          icon={DollarSign}
          color="text-[var(--color-accent)]"
        />
        <StatCard
          label="Spent Today"
          value={`$${data.spent_today.toFixed(2)}`}
          icon={TrendingUp}
          color={pctUsed > 80 ? "text-[var(--color-red)]" : "text-[var(--color-green)]"}
        />
        <StatCard
          label="Remaining"
          value={`$${data.remaining.toFixed(2)}`}
          icon={TrendingDown}
          color="text-[var(--color-yellow)]"
        />
      </div>

      <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-bg-secondary)] p-4">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs text-[var(--color-text-muted)]">Daily usage</span>
          <span className="text-xs font-mono text-[var(--color-text-secondary)]">
            {pctUsed.toFixed(1)}%
          </span>
        </div>
        <div className="h-3 bg-[var(--color-bg-primary)] rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all ${
              pctUsed > 80 ? "bg-[var(--color-red)]" : pctUsed > 50 ? "bg-[var(--color-yellow)]" : "bg-[var(--color-green)]"
            }`}
            style={{ width: `${Math.min(pctUsed, 100)}%` }}
          />
        </div>
      </div>

      {data.hourly_spend.length > 0 && (
        <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-bg-secondary)] p-4">
          <h2 className="text-sm font-semibold text-[var(--color-text-secondary)] uppercase tracking-wider mb-4">
            Hourly Spend
          </h2>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={data.hourly_spend}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="hour" tick={{ fontSize: 11, fill: "#94a3b8" }} />
              <YAxis tick={{ fontSize: 11, fill: "#94a3b8" }} tickFormatter={(v) => `$${v}`} />
              <Tooltip
                contentStyle={{
                  background: "#1e293b",
                  border: "1px solid #475569",
                  borderRadius: 8,
                  fontSize: 12,
                }}
                labelStyle={{ color: "#e2e8f0" }}
                formatter={(value: number) => [`$${value.toFixed(4)}`, "Cost"]}
              />
              <Bar dataKey="cost" radius={[4, 4, 0, 0]}>
                {data.hourly_spend.map((_, i) => (
                  <Cell key={i} fill={i === data.hourly_spend.length - 1 ? "#00d4ff" : "#8b5cf6"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
