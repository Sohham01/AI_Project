/**
 * Read-only local API client for /api/demo with response caching and clean error extraction.
 */

const cache = new Map();

export function clearApiCache() {
  cache.clear();
}

export async function apiGet(path, { useCache = true } = {}) {
  if (useCache && cache.has(path)) {
    return cache.get(path);
  }

  let response;
  try {
    response = await fetch(`/api/demo${path}`, {
      headers: { Accept: "application/json" },
    });
  } catch (networkErr) {
    throw new Error(
      "Could not reach the local FastAPI demo server (/api/demo). Verify that uvicorn is running on this origin."
    );
  }

  if (!response.ok) {
    let detail = `API request failed with HTTP ${response.status}`;
    try {
      const body = await response.json();
      if (typeof body?.detail === "string" && body.detail.trim()) {
        detail = body.detail;
      }
    } catch {
      // Keep status-code fallback if body is non-JSON.
    }
    const err = new Error(detail);
    err.status = response.status;
    throw err;
  }

  const payload = await response.json();
  if (useCache) {
    cache.set(path, payload);
  }
  return payload;
}

export const api = {
  health: () => apiGet("/health", { useCache: false }),
  summary: () => apiGet("/summary"),
  issues: () => apiGet("/issues"),
  topics: () => apiGet("/topics"),
  sentiment: () => apiGet("/sentiment"),
  responses: () => apiGet("/responses"),
  resolution: () => apiGet("/resolution"),
  models: () => apiGet("/models"),
  discussions: (params = {}) => {
    const sp = new URLSearchParams();
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined && v !== null && String(v).trim() !== "") {
        sp.set(k, String(v));
      }
    }
    return apiGet(`/discussions?${sp.toString()}`, { useCache: false });
  },
  discussionDetail: (postId) =>
    apiGet(`/discussions/${encodeURIComponent(postId)}`, { useCache: true }),
};
