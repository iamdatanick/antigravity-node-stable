import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "./client";

/** Response from orchestrator /budget/history */
export interface BudgetHistory {
  current_spend: number;
  max_daily: number;
  currency: string;
  hourly_spend: number[];
}

/** Response from budget-proxy /health (via nginx /api/budget/health) */
export interface BudgetHealth {
  status: string;
  daily_spend_usd: number;
  daily_budget_usd: number;
  remaining_usd: number;
}

/** Normalized budget data for UI consumption */
export interface BudgetData {
  daily_budget: number;
  spent_today: number;
  remaining: number;
  currency: string;
  hourly_spend: { hour: string; cost: number }[];
}

function normalizeHourly(hourly: number[]): { hour: string; cost: number }[] {
  return hourly.map((cost, i) => ({
    hour: `${String(i).padStart(2, "0")}:00`,
    cost,
  }));
}

export function useBudget() {
  return useQuery<BudgetData>({
    queryKey: ["budget"],
    queryFn: async () => {
      // Try orchestrator /budget/history first, fall back to budget-proxy /health
      try {
        const history = await apiFetch<BudgetHistory>("/budget/history");
        return {
          daily_budget: history.max_daily,
          spent_today: history.current_spend,
          remaining: history.max_daily - history.current_spend,
          currency: history.currency || "USD",
          hourly_spend: normalizeHourly(history.hourly_spend || []),
        };
      } catch {
        // Fallback: budget-proxy health endpoint
        const health = await apiFetch<BudgetHealth>("/budget/health");
        return {
          daily_budget: health.daily_budget_usd,
          spent_today: health.daily_spend_usd,
          remaining: health.remaining_usd,
          currency: "USD",
          hourly_spend: [],
        };
      }
    },
    refetchInterval: 30_000,
  });
}
