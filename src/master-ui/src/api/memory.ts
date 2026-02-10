import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "./client";

export interface MemoryEntry {
  event_id: number;
  tenant_id: string;
  timestamp: string;
  session_id: string;
  actor: string;
  action_type: string;
  content: string;
}

export interface MemoryResponse {
  entries: MemoryEntry[];
  total: number;
  limit: number;
  offset: number;
}

export function useMemory(page = 1, pageSize = 25, search?: string) {
  const offset = (page - 1) * pageSize;
  const params = new URLSearchParams({
    limit: String(pageSize),
    offset: String(offset),
  });
  if (search) params.set("search", search);

  return useQuery<MemoryResponse>({
    queryKey: ["memory", page, pageSize, search],
    queryFn: () => apiFetch(`/memory?${params}`),
  });
}
