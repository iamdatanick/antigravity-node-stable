import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "./client";

/** A single provider key entry from /api/settings/keys */
export interface ApiKeyEntry {
  provider: string;
  masked_key: string;
  configured: boolean;
}

/** Response from GET /api/settings/keys */
export interface ApiKeyListResponse {
  keys: ApiKeyEntry[];
}

/** Request body for POST /api/settings/keys */
export interface SaveApiKeyRequest {
  provider: string;
  api_key: string;
}

/** Fetch all configured LLM provider API keys (masked) */
export function useApiKeys() {
  return useQuery<ApiKeyListResponse>({
    queryKey: ["api-keys"],
    queryFn: () => apiFetch("/api/settings/keys"),
    staleTime: 30_000,
  });
}

/** Save an API key for a given LLM provider */
export function useSaveApiKey() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: SaveApiKeyRequest) =>
      apiFetch("/api/settings/keys", {
        method: "POST",
        body: JSON.stringify(data),
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["api-keys"] }),
  });
}

/** Delete an API key for a given LLM provider */
export function useDeleteApiKey() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (provider: string) =>
      apiFetch(`/api/settings/keys/${provider}`, { method: "DELETE" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["api-keys"] }),
  });
}
