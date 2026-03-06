// ---------------------------------------------------------------------------
// ARA API Type Definitions
// Mirrors the Python FastAPI schemas so the frontend stays in sync.
// ---------------------------------------------------------------------------

// ---- Chat -----------------------------------------------------------------

export type MessageRole = "user" | "assistant" | "system";

export interface MessageItem {
  role: MessageRole;
  content: string;
  timestamp?: string;
}

export interface ChatRequest {
  message: string;
  conversation_id?: string;
  history?: MessageItem[];
  stream?: boolean;
  max_steps?: number;
}

export interface ChatResponse {
  response: string;
  conversation_id: string;
  intent: string;
  agent: string;
  confidence: number;
  elapsed_ms: number;
  tool_calls: Record<string, unknown>[];
  tool_results: Record<string, unknown>[];
  error: string;
}

export interface StreamChunk {
  type: "token" | "done" | "error" | "metadata";
  content: string;
  metadata: Record<string, unknown>;
}

// ---- WebSocket ------------------------------------------------------------

export interface WSMessage {
  type: "chat" | "ping" | "cancel" | "status";
  content: string;
  conversation_id: string;
  metadata: Record<string, unknown>;
}

export interface WSResponse {
  type: "token" | "done" | "error" | "pong" | "status";
  content: string;
  conversation_id: string;
  metadata: Record<string, unknown>;
}

// ---- Memory ---------------------------------------------------------------

export interface MemorySearchRequest {
  query: string;
  top_k?: number;
  source_filter?: string;
  date_from?: string;
  date_to?: string;
}

export interface MemoryChunkItem {
  chunk_id: string;
  text: string;
  source: string;
  source_file: string;
  date?: string;
  section_title: string;
  score: number;
  retrieval_method: string;
}

export interface MemorySearchResponse {
  results: MemoryChunkItem[];
  query: string;
  total_candidates: number;
  retrieval_time_ms: number;
}

export interface MemoryStatsResponse {
  total_chunks: number;
  sources: Record<string, number>;
  vector_db_path: string;
  graph_entities: number;
  graph_relations: number;
}

// ---- Voice ----------------------------------------------------------------

export interface VoiceStatusResponse {
  state: string;
  wake_word_enabled: boolean;
  hotkey_enabled: boolean;
  stt_model: string;
  tts_model: string;
  tts_voice: string;
}

export interface VoiceControlRequest {
  action: "start" | "stop" | "toggle_wake_word" | "toggle_hotkey";
}

export interface VoiceControlResponse {
  action: string;
  success: boolean;
  state: string;
  message: string;
}

// ---- Models ---------------------------------------------------------------

export interface ModelItem {
  role: string;
  name: string;
  location: "gpu" | "cpu";
  vram_gb: number;
  ram_mb: number;
  loaded: boolean;
  description: string;
}

export interface VRAMUsageResponse {
  total_gb: number;
  used_gb: number;
  free_gb: number;
  models: ModelItem[];
}

export interface ModelSwapRequest {
  target_model: string;
  reason?: string;
}

export interface ModelSwapResponse {
  previous_model: string;
  current_model: string;
  success: boolean;
  elapsed_ms: number;
  message: string;
}

// ---- Settings -------------------------------------------------------------

export interface PersonalitySettings {
  humor_dial: number;
  voice_identity: string;
  auto_dial_down_on_distress: boolean;
  distress_dial_reduction: number;
}

export interface PersonalityUpdateRequest {
  humor_dial?: number;
  auto_dial_down_on_distress?: boolean;
  distress_dial_reduction?: number;
}

export interface DesktopSettings {
  enabled: boolean;
  require_confirmation: boolean;
  command_timeout_seconds: number;
}

export interface DesktopUpdateRequest {
  enabled?: boolean;
  require_confirmation?: boolean;
}

export interface InternetSettings {
  internet_enabled: boolean;
  searxng_url: string;
  web_cache_ttl_s: number;
}

export interface InternetUpdateRequest {
  internet_enabled?: boolean;
}

export interface SettingsResponse {
  personality: PersonalitySettings;
  desktop: DesktopSettings;
  internet: InternetSettings;
}

// ---- Conversations --------------------------------------------------------

export interface ConversationItem {
  conversation_id: string;
  started_at?: string;
  ended_at?: string;
  turn_count: number;
  summary: string;
  is_active: boolean;
}

export interface ConversationDetailResponse {
  conversation_id: string;
  started_at?: string;
  ended_at?: string;
  turns: MessageItem[];
  summary: string;
}

export interface ConversationListResponse {
  conversations: ConversationItem[];
  total: number;
}

// ---- Common ---------------------------------------------------------------

export interface ErrorResponse {
  error: string;
  detail: string;
  component: string;
  operation: string;
  status_code: number;
}

export interface HealthResponse {
  status: string;
  version: string;
  uptime_s: number;
  models_loaded: string[];
}
