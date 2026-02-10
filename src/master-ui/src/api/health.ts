import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "./client";

/** Individual service check from /health levels[].checks[] */
export interface ServiceCheck {
  name: string;
  healthy: boolean;
  error: string | null;
}

/** A single layer from /health levels[] */
export interface HealthLevel {
  level: string;
  name: string;
  checks: ServiceCheck[];
}

/** Top-level /health response */
export interface HealthData {
  status: "healthy" | "degraded";
  levels: HealthLevel[];
}

/** Capabilities response from /capabilities */
export interface CapabilitiesData {
  node: string;
  protocols: string[];
  endpoints: Record<string, string>;
  mcp_servers: Record<string, { transport: string; url: string }>;
  memory: Record<string, string>;
  budget: { proxy: string; max_daily: string; model: string };
}

export function useHealth() {
  return useQuery<HealthData>({
    queryKey: ["health"],
    queryFn: () => apiFetch("/health", { acceptStatuses: [503] }),
    refetchInterval: 10_000,
    retry: 2,
    retryDelay: 3000,
  });
}

export function useCapabilities() {
  return useQuery<CapabilitiesData>({
    queryKey: ["capabilities"],
    queryFn: () => apiFetch("/capabilities"),
    staleTime: 60_000,
  });
}

/** Derive total service count from health levels */
export function countServices(data: HealthData | undefined) {
  if (!data?.levels) return { total: 0, healthy: 0, unhealthy: 0 };
  let total = 0;
  let healthy = 0;
  for (const level of data.levels) {
    for (const check of level.checks) {
      total++;
      if (check.healthy) healthy++;
    }
  }
  return { total, healthy, unhealthy: total - healthy };
}
