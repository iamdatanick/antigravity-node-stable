import { useState } from "react";
import { useCapabilities } from "../api/health";
import type { ApiKeyEntry } from "../api/settings";
import { useApiKeys, useSaveApiKey, useDeleteApiKey } from "../api/settings";
import { useTools } from "../api/tools";
import {
  Activity,
  Check,
  Database,
  DollarSign,
  Globe,
  Key,
  Save,
  Settings as SettingsIcon,
  Trash2,
  Wrench,
  X,
} from "lucide-react";

interface ProviderConfig {
  id: string;
  label: string;
  placeholder: string;
}

const PROVIDERS: ProviderConfig[] = [
  { id: "openai", label: "OpenAI", placeholder: "sk-..." },
  { id: "anthropic", label: "Anthropic", placeholder: "sk-ant-..." },
  { id: "google", label: "Google AI", placeholder: "AIza..." },
  { id: "mistral", label: "Mistral", placeholder: "..." },
];

interface ProviderKeyRowProps {
  provider: ProviderConfig;
  entry: ApiKeyEntry | undefined;
  inputVal: string;
  onInputChange: (value: string) => void;
  onSave: () => void;
  onDelete: () => void;
  savePending: boolean;
  deletePending: boolean;
}

function ProviderKeyRow({
  provider,
  entry,
  inputVal,
  onInputChange,
  onSave,
  onDelete,
  savePending,
  deletePending,
}: ProviderKeyRowProps) {
  const configured = entry?.configured ?? false;

  return (
    <div className="flex items-center gap-3 bg-[var(--color-bg-primary)] rounded-lg px-3 py-2">
      <div className="w-24 shrink-0">
        <div className="text-xs font-medium text-[var(--color-text-primary)]">{provider.label}</div>
        <div className="flex items-center gap-1 mt-0.5">
          {configured ? (
            <span className="flex items-center gap-0.5 text-[10px] text-green-500">
              <Check size={10} /> Active
            </span>
          ) : (
            <span className="flex items-center gap-0.5 text-[10px] text-[var(--color-text-muted)]">
              <X size={10} /> Not set
            </span>
          )}
        </div>
      </div>
      {configured && entry?.masked_key ? (
        <span className="font-mono text-xs text-[var(--color-text-muted)] flex-1">{entry.masked_key}</span>
      ) : null}
      <input
        type="password"
        placeholder={provider.placeholder}
        value={inputVal}
        onChange={(e) => onInputChange(e.target.value)}
        className="flex-1 min-w-0 px-2 py-1 text-xs rounded border border-[var(--color-border)] bg-[var(--color-bg-secondary)] text-[var(--color-text-primary)] font-mono placeholder:text-[var(--color-text-muted)]"
      />
      <button
        onClick={onSave}
        disabled={!inputVal.trim() || savePending}
        className="p-1.5 rounded hover:bg-[var(--color-accent-dim)] text-[var(--color-accent)] disabled:opacity-30 disabled:cursor-not-allowed"
        title="Save key"
      >
        <Save size={14} />
      </button>
      {configured && (
        <button
          onClick={onDelete}
          disabled={deletePending}
          className="p-1.5 rounded hover:bg-red-500/10 text-red-500 disabled:opacity-30 disabled:cursor-not-allowed"
          title="Delete key"
        >
          <Trash2 size={14} />
        </button>
      )}
    </div>
  );
}

export default function Settings() {
  const { data: capabilities, isLoading: capLoading } = useCapabilities();
  const { data: toolsData, isLoading: toolsLoading } = useTools();
  const { data: keysData, isLoading: keysLoading } = useApiKeys();
  const saveKey = useSaveApiKey();
  const deleteKey = useDeleteApiKey();
  const [keyInputs, setKeyInputs] = useState<Record<string, string>>({});

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

      {/* LLM Providers — API Key Management */}
      <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-bg-secondary)] p-4">
        <h2 className="text-sm font-semibold text-[var(--color-text-secondary)] uppercase tracking-wider mb-3 flex items-center gap-2">
          <Key size={14} />
          LLM Providers
        </h2>
        <p className="text-xs text-[var(--color-text-muted)] mb-4">
          API keys are stored securely in OpenBao vault. Enter keys for external LLM providers.
        </p>
        {keysLoading ? (
          <Activity className="animate-spin text-[var(--color-accent)]" size={20} />
        ) : (
          <div className="space-y-3">
            {PROVIDERS.map((prov) => (
              <ProviderKeyRow
                key={prov.id}
                provider={prov}
                entry={keysData?.keys.find((k) => k.provider === prov.id)}
                inputVal={keyInputs[prov.id] ?? ""}
                onInputChange={(val) => setKeyInputs((prev) => ({ ...prev, [prov.id]: val }))}
                onSave={() => {
                  const val = (keyInputs[prov.id] ?? "").trim();
                  if (!val) return;
                  saveKey.mutate(
                    { provider: prov.id, api_key: val },
                    { onSuccess: () => setKeyInputs((prev) => ({ ...prev, [prov.id]: "" })) },
                  );
                }}
                onDelete={() => deleteKey.mutate(prov.id)}
                savePending={saveKey.isPending}
                deletePending={deleteKey.isPending}
              />
            ))}
          </div>
        )}
      </div>

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
