import { useCapabilities } from "../api/health";
import { useTools } from "../api/tools";
import { Settings as SettingsIcon, Activity, Wrench, Globe, Database, DollarSign } from "lucide-react";

export default function Settings() {
  const { data: capabilities, isLoading: capLoading } = useCapabilities();
  const { data: toolsData, isLoading: toolsLoading } = useTools();

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-bold text-[var(--color-text-primary)] flex items-center gap-2">
          <SettingsIcon size={20} />
          Settings
        </h1>
        <p className="text-sm text-[var(--color-text-muted)]">
          System configuration, agent manifest, and MCP gateway status
        </p>
      </div>

      {/* Node info */}
      {capLoading ? (
        <div className="flex items-center justify-center h-32">
          <Activity className="animate-spin text-[var(--color-accent)]" size={20} />
        </div>
      ) : capabilities ? (
        <div className="space-y-4">
          <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-bg-secondary)] p-4">
            <h2 className="text-sm font-semibold text-[var(--color-text-secondary)] uppercase tracking-wider mb-3 flex items-center gap-2">
              <Globe size={14} />
              Node
            </h2>
            <div className="text-lg font-bold text-[var(--color-text-primary)] mb-2">{capabilities.node}</div>
            <div className="flex flex-wrap gap-2">
              {capabilities.protocols.map((p) => (
                <span
                  key={p}
                  className="px-2 py-0.5 rounded bg-[var(--color-accent-dim)] text-[var(--color-accent)] text-xs font-medium"
                >
                  {p}
                </span>
              ))}
            </div>
          </div>

          {/* Endpoints */}
          <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-bg-secondary)] p-4">
            <h2 className="text-sm font-semibold text-[var(--color-text-secondary)] uppercase tracking-wider mb-3">
              Endpoints
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              {Object.entries(capabilities.endpoints).map(([key, val]) => (
                <div key={key} className="flex items-center justify-between text-xs bg-[var(--color-bg-primary)] rounded-lg px-3 py-2">
                  <span className="font-mono text-[var(--color-text-secondary)]">{key}</span>
                  <span className="font-mono text-[var(--color-text-muted)]">{val}</span>
                </div>
              ))}
            </div>
          </div>

          {/* MCP Servers */}
          <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-bg-secondary)] p-4">
            <h2 className="text-sm font-semibold text-[var(--color-text-secondary)] uppercase tracking-wider mb-3 flex items-center gap-2">
              <Wrench size={14} />
              MCP Servers
            </h2>
            <div className="space-y-2">
              {Object.entries(capabilities.mcp_servers).map(([name, info]) => (
                <div key={name} className="flex items-center justify-between text-xs bg-[var(--color-bg-primary)] rounded-lg px-3 py-2">
                  <span className="font-medium text-[var(--color-text-primary)]">{name}</span>
                  <div className="flex items-center gap-3">
                    <span className="px-1.5 py-0.5 rounded bg-[var(--color-purple-dim)] text-[var(--color-purple)] font-mono">
                      {info.transport}
                    </span>
                    <span className="font-mono text-[var(--color-text-muted)]">{info.url}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Memory stores */}
          <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-bg-secondary)] p-4">
            <h2 className="text-sm font-semibold text-[var(--color-text-secondary)] uppercase tracking-wider mb-3 flex items-center gap-2">
              <Database size={14} />
              Memory Stores
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              {Object.entries(capabilities.memory).map(([key, val]) => (
                <div key={key} className="flex items-center justify-between text-xs bg-[var(--color-bg-primary)] rounded-lg px-3 py-2">
                  <span className="font-medium text-[var(--color-text-primary)]">{key}</span>
                  <span className="font-mono text-[var(--color-text-muted)]">{val}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Budget config */}
          <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-bg-secondary)] p-4">
            <h2 className="text-sm font-semibold text-[var(--color-text-secondary)] uppercase tracking-wider mb-3 flex items-center gap-2">
              <DollarSign size={14} />
              Budget Config
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
              {Object.entries(capabilities.budget).map(([key, val]) => (
                <div key={key} className="text-xs bg-[var(--color-bg-primary)] rounded-lg px-3 py-2">
                  <div className="text-[var(--color-text-muted)] mb-0.5">{key}</div>
                  <div className="font-mono text-[var(--color-text-primary)]">{val}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      ) : (
        <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-bg-secondary)] h-32 flex items-center justify-center">
          <p className="text-sm text-[var(--color-text-muted)]">Cannot load capabilities</p>
        </div>
      )}

      {/* Tools */}
      <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-bg-secondary)] p-4">
        <h2 className="text-sm font-semibold text-[var(--color-text-secondary)] uppercase tracking-wider mb-3 flex items-center gap-2">
          <Wrench size={14} />
          Available Tools {toolsData && `(${toolsData.total})`}
        </h2>
        {toolsLoading ? (
          <Activity className="animate-spin text-[var(--color-accent)]" size={20} />
        ) : toolsData && toolsData.tools.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
            {toolsData.tools.map((tool) => (
              <div key={tool.name} className="text-xs bg-[var(--color-bg-primary)] rounded-lg px-3 py-2">
                <div className="font-medium text-[var(--color-text-primary)]">{tool.name}</div>
                <div className="text-[var(--color-text-muted)] mt-0.5">
                  {tool.server}
                  {tool.description && ` — ${tool.description}`}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-xs text-[var(--color-text-muted)]">No tools loaded</p>
        )}
      </div>
    </div>
  );
}
