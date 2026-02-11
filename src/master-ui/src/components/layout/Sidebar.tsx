import { NavLink } from "react-router";
import {
  LayoutDashboard,
  MessageSquare,
  Sparkles,
  ScrollText,
  Database,
  Terminal,
  Workflow,
  DollarSign,
  Server,
  Settings,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";
import { useState } from "react";

const NAV = [
  { to: "/", icon: LayoutDashboard, label: "Dashboard", available: true },
  { to: "/hybrid", icon: Sparkles, label: "Hybrid Chat", available: true },
  { to: "/chat", icon: MessageSquare, label: "Legacy Chat", available: true },
  { to: "/logs", icon: ScrollText, label: "Logs", available: false },
  { to: "/budget", icon: DollarSign, label: "Budget", available: true },
  { to: "/services", icon: Server, label: "Services", available: true },
  { to: "/settings", icon: Settings, label: "Settings", available: true },
] as const;

export default function Sidebar() {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <aside
      className={`flex flex-col border-r border-[var(--color-border)] bg-[var(--color-bg-secondary)] transition-all duration-200 ${
        collapsed ? "w-16" : "w-56"
      }`}
    >
      <div className="flex items-center gap-3 px-4 py-5 border-b border-[var(--color-border)]">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-[var(--color-accent)] to-[var(--color-purple)] flex items-center justify-center text-white font-bold text-sm shrink-0">
          AG
        </div>
        {!collapsed && (
          <div className="overflow-hidden">
            <div className="text-sm font-bold text-[var(--color-text-primary)] truncate">
              Antigravity
            </div>
            <div className="text-[10px] text-[var(--color-text-muted)]">v14.1 — Phoenix</div>
          </div>
        )}
      </div>

      <nav className="flex-1 py-3 space-y-0.5 px-2 overflow-y-auto">
        {NAV.map(({ to, icon: Icon, label, available }) => (
          <NavLink
            key={to}
            to={to}
            end={to === "/"}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors ${
                !available
                  ? "opacity-30 cursor-default text-[var(--color-text-muted)]"
                  : isActive
                    ? "bg-[var(--color-accent-dim)] text-[var(--color-accent)] font-medium"
                    : "text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-tertiary)] hover:text-[var(--color-text-primary)]"
              }`
            }
          >
            <Icon size={18} className="shrink-0" />
            {!collapsed && <span className="truncate">{label}</span>}
          </NavLink>
        ))}
      </nav>

      <button
        onClick={() => setCollapsed((c) => !c)}
        className="flex items-center justify-center py-3 border-t border-[var(--color-border)] text-[var(--color-text-muted)] hover:text-[var(--color-text-primary)] transition-colors"
      >
        {collapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
      </button>
    </aside>
  );
}
