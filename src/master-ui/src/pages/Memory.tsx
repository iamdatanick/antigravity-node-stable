import { useState } from "react";
import { useMemory } from "../api/memory";
import {
  Database,
  ChevronLeft,
  ChevronRight,
  Activity,
  Search,
} from "lucide-react";

export default function Memory() {
  const [page, setPage] = useState(1);
  const pageSize = 25;
  const [search, setSearch] = useState("");
  const { data, isLoading, isError } = useMemory(page, pageSize, search || undefined);

  const totalPages = data ? Math.ceil(data.total / pageSize) : 1;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-[var(--color-text-primary)] flex items-center gap-2">
            <Database size={20} />
            Memory Browser
          </h1>
          <p className="text-sm text-[var(--color-text-muted)]">
            Episodic, semantic, and procedural memory traces
          </p>
        </div>
        <div className="relative">
          <Search
            size={14}
            className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--color-text-muted)]"
          />
          <input
            type="text"
            placeholder="Search memories..."
            value={search}
            onChange={(e) => {
              setSearch(e.target.value);
              setPage(1);
            }}
            className="pl-8 pr-3 py-1.5 text-sm bg-[var(--color-bg-secondary)] border border-[var(--color-border)] rounded-lg text-[var(--color-text-primary)] placeholder:text-[var(--color-text-muted)] focus:outline-none focus:border-[var(--color-accent)]"
          />
        </div>
      </div>

      {isLoading ? (
        <div className="flex items-center justify-center h-64">
          <Activity className="animate-spin text-[var(--color-accent)]" size={24} />
        </div>
      ) : isError || !data ? (
        <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-bg-secondary)] h-64 flex items-center justify-center">
          <p className="text-sm text-[var(--color-text-muted)]">
            Cannot load memory — make sure the stack is running
          </p>
        </div>
      ) : (
        <>
          <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-bg-secondary)] overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-[var(--color-border)] bg-[var(--color-bg-primary)]">
                  <th className="text-left px-4 py-2.5 text-xs font-semibold text-[var(--color-text-muted)] uppercase tracking-wider">
                    Timestamp
                  </th>
                  <th className="text-left px-4 py-2.5 text-xs font-semibold text-[var(--color-text-muted)] uppercase tracking-wider">
                    Actor
                  </th>
                  <th className="text-left px-4 py-2.5 text-xs font-semibold text-[var(--color-text-muted)] uppercase tracking-wider">
                    Action
                  </th>
                  <th className="text-left px-4 py-2.5 text-xs font-semibold text-[var(--color-text-muted)] uppercase tracking-wider">
                    Content
                  </th>
                </tr>
              </thead>
              <tbody>
                {data.entries.length === 0 ? (
                  <tr>
                    <td colSpan={4} className="px-4 py-8 text-center text-[var(--color-text-muted)]">
                      No memory entries found
                    </td>
                  </tr>
                ) : (
                  data.entries.map((entry) => (
                    <tr
                      key={entry.event_id}
                      className="border-b border-[var(--color-border)] hover:bg-[var(--color-bg-primary)] transition-colors"
                    >
                      <td className="px-4 py-2.5 text-[var(--color-text-muted)] font-mono text-xs whitespace-nowrap">
                        {new Date(entry.timestamp).toLocaleString()}
                      </td>
                      <td className="px-4 py-2.5">
                        <span className="px-2 py-0.5 rounded bg-[var(--color-purple-dim)] text-[var(--color-purple)] text-xs font-medium">
                          {entry.actor}
                        </span>
                      </td>
                      <td className="px-4 py-2.5">
                        <span className="px-2 py-0.5 rounded bg-[var(--color-accent-dim)] text-[var(--color-accent)] text-xs font-medium">
                          {entry.action_type}
                        </span>
                      </td>
                      <td className="px-4 py-2.5 text-[var(--color-text-secondary)] max-w-md truncate">
                        {entry.content}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>

          <div className="flex items-center justify-between">
            <span className="text-xs text-[var(--color-text-muted)]">
              {data.total} entries total — Page {page} of {totalPages}
            </span>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={page === 1}
                className="p-1.5 rounded border border-[var(--color-border)] text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] disabled:opacity-30 transition-colors"
              >
                <ChevronLeft size={14} />
              </button>
              <button
                onClick={() => setPage((p) => p + 1)}
                disabled={page >= totalPages}
                className="p-1.5 rounded border border-[var(--color-border)] text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] disabled:opacity-30 transition-colors"
              >
                <ChevronRight size={14} />
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
