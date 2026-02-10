import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "./client";

export interface WorkflowNodeRaw {
  id: string;
  name: string;
  type: string;
  phase: string;
  dependencies: string[];
}

export interface WorkflowRaw {
  name: string;
  phase: string;
  started_at: string;
  finished_at: string | null;
  nodes: WorkflowNodeRaw[];
}

export interface WorkflowsResponse {
  workflows: WorkflowRaw[];
}

/** Cytoscape-compatible node */
export interface CyNode {
  id: string;
  label: string;
  type: string;
  status: string;
}

/** Cytoscape-compatible edge */
export interface CyEdge {
  source: string;
  target: string;
}

/** Transformed data for Cytoscape DAG */
export interface WorkflowData {
  nodes: CyNode[];
  edges: CyEdge[];
}

/** Map Argo phase to simple status */
function phaseToStatus(phase: string): string {
  switch (phase) {
    case "Running": return "active";
    case "Succeeded": return "idle";
    case "Failed": return "error";
    case "Pending": return "pending";
    default: return "idle";
  }
}

/** Map Argo node type to visual type */
function nodeTypeToVisual(type: string): string {
  switch (type) {
    case "Pod": return "agent";
    case "DAG": return "gateway";
    case "Steps": return "tool";
    default: return "model";
  }
}

/** Transform raw Argo workflow response to Cytoscape-compatible format */
function transformWorkflows(raw: WorkflowsResponse): WorkflowData {
  const nodes: CyNode[] = [];
  const edges: CyEdge[] = [];

  for (const wf of raw.workflows) {
    for (const node of wf.nodes) {
      nodes.push({
        id: node.id,
        label: node.name,
        type: nodeTypeToVisual(node.type),
        status: phaseToStatus(node.phase),
      });
      for (const dep of node.dependencies || []) {
        edges.push({ source: dep, target: node.id });
      }
    }
  }

  return { nodes, edges };
}

export function useWorkflows() {
  return useQuery<WorkflowData>({
    queryKey: ["workflows"],
    queryFn: async () => {
      const raw = await apiFetch<WorkflowsResponse>("/workflows");
      return transformWorkflows(raw);
    },
    staleTime: 30_000,
  });
}
