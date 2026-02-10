const BASE = "/api";

/** Retry fetch with exponential backoff for transient errors */
async function fetchWithRetry(
  url: string,
  init?: RequestInit,
  retries = 2,
  delay = 1000,
  acceptStatuses?: number[],
): Promise<Response> {
  for (let i = 0; i <= retries; i++) {
    const res = await fetch(url, init);
    const accepted = res.ok || acceptStatuses?.includes(res.status);
    if (accepted || res.status === 400 || res.status === 401 || res.status === 404 || i === retries) {
      return res;
    }
    // Retry on 429, 502, 503, 504
    if (res.status === 429 || res.status >= 500) {
      await new Promise((r) => setTimeout(r, delay * Math.pow(2, i)));
      continue;
    }
    return res;
  }
  // Unreachable, but satisfies TypeScript
  return fetch(url, init);
}

interface ApiFetchOptions extends RequestInit {
  /** Extra HTTP status codes to treat as success (e.g. 503 for health endpoint) */
  acceptStatuses?: number[];
}

export async function apiFetch<T>(path: string, init?: ApiFetchOptions): Promise<T> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };

  // Add tenant header for multi-tenant isolation
  headers["X-Tenant-Id"] = "default";

  const { acceptStatuses, ...fetchInit } = init || {};

  const res = await fetchWithRetry(
    `${BASE}${path}`,
    { ...fetchInit, headers: { ...headers, ...fetchInit?.headers } },
    2,
    1000,
    acceptStatuses,
  );

  if (!res.ok && !acceptStatuses?.includes(res.status)) {
    const body = await res.text().catch(() => "");
    let detail = `${res.status} ${res.statusText}`;
    try {
      const json = JSON.parse(body);
      if (json.detail) detail = json.detail;
    } catch { /* use status text */ }

    if (res.status === 429) throw new Error("Rate limit exceeded — try again shortly");
    throw new Error(detail);
  }
  return res.json();
}

export async function* streamChat(
  messages: { role: string; content: string }[],
  model = "gpt-4o",
  signal?: AbortSignal,
) {
  const res = await fetch(`${BASE}/v1/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Tenant-Id": "default",
    },
    body: JSON.stringify({ model, messages, stream: true }),
    signal,
  });

  if (res.status === 429) throw new Error("Rate limit exceeded — try again shortly");

  if (!res.ok) {
    const body = await res.text().catch(() => "");
    try {
      const json = JSON.parse(body);
      throw new Error(json.detail || `${res.status} ${res.statusText}`);
    } catch (e) {
      if (e instanceof Error && e.message !== body) throw e;
      throw new Error(`${res.status} ${res.statusText}`);
    }
  }

  // Handle non-streaming response (budget exhaustion returns plain JSON)
  const contentType = res.headers.get("content-type") || "";
  if (!contentType.includes("text/event-stream")) {
    const json = await res.json();
    const content = json.choices?.[0]?.message?.content;
    if (content) yield content;
    return;
  }

  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const lines = buf.split("\n");
    buf = lines.pop()!;
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed.startsWith("data: ") || trimmed === "data: [DONE]") continue;
      try {
        const json = JSON.parse(trimmed.slice(6));
        const delta = json.choices?.[0]?.delta?.content;
        if (delta) yield delta;
      } catch { /* skip malformed chunks */ }
    }
  }
}
