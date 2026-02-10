import { useEffect, useRef } from "react";
import cytoscape from "cytoscape";
import { useWorkflows } from "../api/workflows";
import { Workflow, Activity } from "lucide-react";

const STATUS_COLORS: Record<string, string> = {
  active: "#22c55e",
  idle: "#6b7280",
  error: "#ef4444",
  pending: "#eab308",
};

const TYPE_SHAPES: Record<string, cytoscape.Css.NodeShape> = {
  agent: "round-rectangle",
  tool: "diamond",
  model: "ellipse",
  gateway: "hexagon",
};

export default function Workflows() {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<cytoscape.Core | null>(null);
  const { data, isLoading, isError } = useWorkflows();

  useEffect(() => {
    if (!containerRef.current || !data) return;

    const elements: cytoscape.ElementDefinition[] = [
      ...data.nodes.map((n) => ({
        data: { id: n.id, label: n.label, type: n.type, status: n.status },
      })),
      ...data.edges.map((e) => ({
        data: { source: e.source, target: e.target },
      })),
    ];

    const cy = cytoscape({
      container: containerRef.current,
      elements,
      style: [
        {
          selector: "node",
          style: {
            label: "data(label)",
            "text-valign": "center",
            "text-halign": "center",
            "font-size": "11px",
            color: "#e2e8f0",
            "text-outline-color": "#0f172a",
            "text-outline-width": 2,
            "background-color": (ele: cytoscape.NodeSingular) =>
              STATUS_COLORS[ele.data("status") || "idle"] || "#6b7280",
            shape: (ele: cytoscape.NodeSingular) =>
              TYPE_SHAPES[ele.data("type") || "agent"] || "ellipse",
            width: 50,
            height: 50,
            "border-width": 2,
            "border-color": "#334155",
          },
        },
        {
          selector: "edge",
          style: {
            width: 2,
            "line-color": "#475569",
            "target-arrow-color": "#475569",
            "target-arrow-shape": "triangle",
            "curve-style": "bezier",
            label: "data(label)",
            "font-size": "9px",
            color: "#94a3b8",
            "text-outline-color": "#0f172a",
            "text-outline-width": 1,
          },
        },
        {
          selector: "node:selected",
          style: {
            "border-color": "#00d4ff",
            "border-width": 3,
          },
        },
      ],
      layout: {
        name: "breadthfirst",
        directed: true,
        spacingFactor: 1.5,
        padding: 30,
      },
    });

    cyRef.current = cy;
    return () => cy.destroy();
  }, [data]);

  return (
    <div className="flex flex-col h-[calc(100vh-8rem)]">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h1 className="text-xl font-bold text-[var(--color-text-primary)] flex items-center gap-2">
            <Workflow size={20} />
            Workflow DAG
          </h1>
          <p className="text-sm text-[var(--color-text-muted)]">
            Agent workflow visualization — click nodes to inspect
          </p>
        </div>
        <div className="flex items-center gap-3 text-xs text-[var(--color-text-muted)]">
          <span className="flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded-sm bg-[#22c55e]" /> Active
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded-sm bg-[#eab308]" /> Pending
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded-sm bg-[#6b7280]" /> Idle
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded-sm bg-[#ef4444]" /> Error
          </span>
        </div>
      </div>

      {isLoading ? (
        <div className="flex-1 flex items-center justify-center">
          <Activity className="animate-spin text-[var(--color-accent)]" size={24} />
        </div>
      ) : isError || !data ? (
        <div className="flex-1 rounded-xl border border-[var(--color-border)] bg-[var(--color-bg-secondary)] flex items-center justify-center">
          <p className="text-sm text-[var(--color-text-muted)]">
            Cannot load workflows — make sure the stack is running
          </p>
        </div>
      ) : (
        <div
          ref={containerRef}
          className="flex-1 rounded-xl border border-[var(--color-border)] bg-[var(--color-bg-primary)]"
        />
      )}
    </div>
  );
}
