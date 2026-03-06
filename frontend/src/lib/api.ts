// ---------------------------------------------------------------------------
// ARA API Client
// Wraps fetch calls to the FastAPI backend.
// ---------------------------------------------------------------------------

import type {
  ChatRequest,
  ChatResponse,
  ConversationDetailResponse,
  ConversationListResponse,
  DesktopSettings,
  DesktopUpdateRequest,
  ErrorResponse,
  HealthResponse,
  InternetSettings,
  InternetUpdateRequest,
  MemorySearchRequest,
  MemorySearchResponse,
  MemoryStatsResponse,
  ModelItem,
  ModelSwapRequest,
  ModelSwapResponse,
  PersonalitySettings,
  PersonalityUpdateRequest,
  SettingsResponse,
  StreamChunk,
  VoiceControlRequest,
  VoiceControlResponse,
  VoiceStatusResponse,
  VRAMUsageResponse,
} from "@/types/api";

// ---------------------------------------------------------------------------
// Base URL
// ---------------------------------------------------------------------------

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

// ---------------------------------------------------------------------------
// Generic request helper
// ---------------------------------------------------------------------------

class ApiError extends Error {
  status: number;
  body: ErrorResponse | null;

  constructor(status: number, message: string, body: ErrorResponse | null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
  }
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE}${path}`;

  const res = await fetch(url, {
    headers: { "Content-Type": "application/json", ...options?.headers },
    ...options,
  });

  if (!res.ok) {
    let body: ErrorResponse | null = null;
    try {
      body = (await res.json()) as ErrorResponse;
    } catch {
      // response body was not JSON – ignore
    }
    throw new ApiError(
      res.status,
      body?.detail ?? body?.error ?? res.statusText,
      body,
    );
  }

  return (await res.json()) as T;
}

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------

export async function fetchHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/api/health");
}

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------

export async function sendChat(req: ChatRequest): Promise<ChatResponse> {
  return request<ChatResponse>("/api/chat", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export async function* streamChat(
  req: ChatRequest,
): AsyncGenerator<StreamChunk> {
  const url = `${API_BASE}/api/chat/stream`;

  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });

  if (!res.ok) {
    let body: ErrorResponse | null = null;
    try {
      body = (await res.json()) as ErrorResponse;
    } catch {
      // ignore
    }
    throw new ApiError(
      res.status,
      body?.detail ?? body?.error ?? res.statusText,
      body,
    );
  }

  const reader = res.body?.getReader();
  if (!reader) {
    throw new Error("Response body is not readable");
  }

  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // SSE format: "data: {...}\n\n"
      const parts = buffer.split("\n\n");
      // The last element is either empty or an incomplete chunk – keep it.
      buffer = parts.pop() ?? "";

      for (const part of parts) {
        const trimmed = part.trim();
        if (!trimmed) continue;

        for (const line of trimmed.split("\n")) {
          if (line.startsWith("data: ")) {
            const json = line.slice(6);
            if (json === "[DONE]") return;
            try {
              yield JSON.parse(json) as StreamChunk;
            } catch {
              // skip malformed JSON lines
            }
          }
        }
      }
    }

    // Process any remaining data in the buffer
    if (buffer.trim()) {
      for (const line of buffer.trim().split("\n")) {
        if (line.startsWith("data: ")) {
          const json = line.slice(6);
          if (json === "[DONE]") return;
          try {
            yield JSON.parse(json) as StreamChunk;
          } catch {
            // skip malformed JSON lines
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

// ---------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------

export async function searchMemory(
  req: MemorySearchRequest,
): Promise<MemorySearchResponse> {
  return request<MemorySearchResponse>("/api/memory/search", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export async function fetchMemoryStats(): Promise<MemoryStatsResponse> {
  return request<MemoryStatsResponse>("/api/memory/stats");
}

// ---------------------------------------------------------------------------
// Conversations
// ---------------------------------------------------------------------------

export async function fetchConversations(
  limit?: number,
): Promise<ConversationListResponse> {
  const params = limit !== undefined ? `?limit=${limit}` : "";
  return request<ConversationListResponse>(`/api/conversations${params}`);
}

export async function fetchConversation(
  id: string,
): Promise<ConversationDetailResponse> {
  return request<ConversationDetailResponse>(
    `/api/conversations/${encodeURIComponent(id)}`,
  );
}

// ---------------------------------------------------------------------------
// Voice
// ---------------------------------------------------------------------------

export async function fetchVoiceStatus(): Promise<VoiceStatusResponse> {
  return request<VoiceStatusResponse>("/api/voice/status");
}

export async function controlVoice(
  req: VoiceControlRequest,
): Promise<VoiceControlResponse> {
  return request<VoiceControlResponse>("/api/voice/control", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

// ---------------------------------------------------------------------------
// Models & VRAM
// ---------------------------------------------------------------------------

export async function fetchModels(): Promise<ModelItem[]> {
  return request<ModelItem[]>("/api/models");
}

export async function fetchVRAM(): Promise<VRAMUsageResponse> {
  return request<VRAMUsageResponse>("/api/models/vram");
}

export async function swapModel(
  req: ModelSwapRequest,
): Promise<ModelSwapResponse> {
  return request<ModelSwapResponse>("/api/models/swap", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

export async function fetchSettings(): Promise<SettingsResponse> {
  return request<SettingsResponse>("/api/settings");
}

export async function updatePersonality(
  req: PersonalityUpdateRequest,
): Promise<PersonalitySettings> {
  return request<PersonalitySettings>("/api/settings/personality", {
    method: "PATCH",
    body: JSON.stringify(req),
  });
}

export async function updateDesktop(
  req: DesktopUpdateRequest,
): Promise<DesktopSettings> {
  return request<DesktopSettings>("/api/settings/desktop", {
    method: "PATCH",
    body: JSON.stringify(req),
  });
}

export async function updateInternet(
  req: InternetUpdateRequest,
): Promise<InternetSettings> {
  return request<InternetSettings>("/api/settings/internet", {
    method: "PATCH",
    body: JSON.stringify(req),
  });
}

// ---------------------------------------------------------------------------
// Re-export the error class for consumers
// ---------------------------------------------------------------------------

export { ApiError };
