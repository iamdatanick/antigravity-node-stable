import { useState, useCallback } from "react";
import { EditorView, keymap } from "@codemirror/view";
import { EditorState } from "@codemirror/state";
import { sql } from "@codemirror/lang-sql";
import { oneDark } from "@codemirror/theme-one-dark";
import { useEffect, useRef } from "react";
import { apiFetch } from "../api/client";
import { Terminal, Play, Activity, AlertTriangle } from "lucide-react";

interface QueryResult {
  columns: string[];
  rows: unknown[][];
  row_count: number;
  truncated?: boolean;
}

export default function Query() {
  const editorRef = useRef<HTMLDivElement>(null);
  const viewRef = useRef<EditorView | null>(null);
  const [result, setResult] = useState<QueryResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const runQuery = useCallback(async () => {
    if (!viewRef.current) return;
    const sqlText = viewRef.current.state.doc.toString().trim();
    if (!sqlText) return;

    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await apiFetch<QueryResult>("/query", {
        method: "POST",
        body: JSON.stringify({ sql: sqlText }),
      });
      setResult(res);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!editorRef.current) return;

    const state = EditorState.create({
      doc: "SELECT * FROM memory_episodic LIMIT 10;",
      extensions: [
        sql(),
        oneDark,
        keymap.of([
          {
            key: "Ctrl-Enter",
            run: () => {
              runQuery();
              return true;
            },
          },
        ]),
        EditorView.theme({
          "&": { height: "200px", fontSize: "13px" },
          ".cm-scroller": { overflow: "auto" },
        }),
      ],
    });

    const view = new EditorView({ state, parent: editorRef.current });
    viewRef.current = view;

    return () => view.destroy();
  }, [runQuery]);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-[var(--color-text-primary)] flex items-center gap-2">
            <Terminal size={20} />
            SQL Console
          </h1>
          <p className="text-sm text-[var(--color-text-muted)]">
            Execute queries against StarRocks — Ctrl+Enter to run
          </p>
        </div>
        <button
          onClick={runQuery}
          disabled={loading}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[var(--color-accent)] text-white text-sm font-medium hover:opacity-90 disabled:opacity-50 transition-opacity"
        >
          {loading ? <Activity size={14} className="animate-spin" /> : <Play size={14} />}
          Run Query
        </button>
      </div>

      <div
        ref={editorRef}
        className="rounded-xl border border-[var(--color-border)] overflow-hidden"
      />

      {error && (
        <div className="flex items-start gap-2 p-3 rounded-lg bg-[var(--color-red-dim)] border border-[var(--color-red)]/20">
          <AlertTriangle size={14} className="text-[var(--color-red)] mt-0.5 shrink-0" />
          <span className="text-sm text-[var(--color-red)]">{error}</span>
        </div>
      )}

      {result && (
        <div className="space-y-2">
          <div className="flex items-center gap-4 text-xs text-[var(--color-text-muted)]">
            <span>{result.row_count} rows</span>
            {result.truncated && (
              <span className="text-[var(--color-yellow)]">Results truncated</span>
            )}
          </div>
          <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-bg-secondary)] overflow-auto max-h-96">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-[var(--color-border)] bg-[var(--color-bg-primary)]">
                  {result.columns.map((col) => (
                    <th
                      key={col}
                      className="text-left px-3 py-2 text-xs font-semibold text-[var(--color-text-muted)] uppercase tracking-wider whitespace-nowrap"
                    >
                      {col}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {result.rows.map((row, i) => (
                  <tr
                    key={i}
                    className="border-b border-[var(--color-border)] hover:bg-[var(--color-bg-primary)] transition-colors"
                  >
                    {row.map((cell, j) => (
                      <td
                        key={j}
                        className="px-3 py-2 text-[var(--color-text-secondary)] whitespace-nowrap font-mono text-xs"
                      >
                        {String(cell ?? "")}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
