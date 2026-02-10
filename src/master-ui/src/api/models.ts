import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "./client";

export interface ModelEntry {
  id: string;
  object: string;
  owned_by: string;
}

export interface ModelsResponse {
  object: string;
  data: ModelEntry[];
}

export function useModels() {
  return useQuery<ModelsResponse>({
    queryKey: ["models"],
    queryFn: () => apiFetch("/v1/models"),
    staleTime: 60_000,
  });
}
