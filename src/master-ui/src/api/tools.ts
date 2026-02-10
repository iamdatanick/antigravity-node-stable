import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "./client";

export interface ToolEntry {
  name: string;
  server: string;
  description?: string;
  status: string | null;
  transport: string | null;
  url: string | null;
}

export interface ToolsResponse {
  tools: ToolEntry[];
  total: number;
}

export function useTools() {
  return useQuery<ToolsResponse>({
    queryKey: ["tools"],
    queryFn: () => apiFetch("/tools"),
    staleTime: 30_000,
  });
}
