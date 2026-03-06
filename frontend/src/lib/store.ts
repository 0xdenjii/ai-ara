import { create } from "zustand";
import type {
  MessageItem,
  ConversationItem,
  ModelItem,
  PersonalitySettings,
  DesktopSettings,
  InternetSettings,
  VoiceStatusResponse,
} from "@/types/api";

// ---------------------------------------------------------------------------
// Global application state
// ---------------------------------------------------------------------------

interface AppState {
  // Chat
  messages: MessageItem[];
  isStreaming: boolean;
  currentConversationId: string;
  addMessage: (msg: MessageItem) => void;
  setMessages: (msgs: MessageItem[]) => void;
  setStreaming: (streaming: boolean) => void;
  setConversationId: (id: string) => void;
  clearChat: () => void;

  // Sidebar
  sidebarOpen: boolean;
  sidebarTab: "conversations" | "memory";
  toggleSidebar: () => void;
  setSidebarTab: (tab: "conversations" | "memory") => void;

  // Conversations
  conversations: ConversationItem[];
  setConversations: (convos: ConversationItem[]) => void;

  // Models
  models: ModelItem[];
  setModels: (models: ModelItem[]) => void;

  // Settings
  personality: PersonalitySettings;
  desktop: DesktopSettings;
  internet: InternetSettings;
  setPersonality: (p: PersonalitySettings) => void;
  setDesktop: (d: DesktopSettings) => void;
  setInternet: (i: InternetSettings) => void;

  // Voice
  voiceStatus: VoiceStatusResponse;
  setVoiceStatus: (v: VoiceStatusResponse) => void;

  // Connection
  isConnected: boolean;
  setConnected: (connected: boolean) => void;
}

export const useAppStore = create<AppState>()((set) => ({
  // -- Chat -----------------------------------------------------------------
  messages: [],
  isStreaming: false,
  currentConversationId: "",

  addMessage: (msg) =>
    set((state) => ({ messages: [...state.messages, msg] })),

  setMessages: (messages) => set({ messages }),

  setStreaming: (isStreaming) => set({ isStreaming }),

  setConversationId: (currentConversationId) => set({ currentConversationId }),

  clearChat: () =>
    set({ messages: [], currentConversationId: "", isStreaming: false }),

  // -- Sidebar --------------------------------------------------------------
  sidebarOpen: true,
  sidebarTab: "conversations",

  toggleSidebar: () =>
    set((state) => ({ sidebarOpen: !state.sidebarOpen })),

  setSidebarTab: (sidebarTab) => set({ sidebarTab }),

  // -- Conversations --------------------------------------------------------
  conversations: [],
  setConversations: (conversations) => set({ conversations }),

  // -- Models ---------------------------------------------------------------
  models: [],
  setModels: (models) => set({ models }),

  // -- Settings -------------------------------------------------------------
  personality: {
    humor_dial: 7,
    voice_identity: "female",
    auto_dial_down_on_distress: true,
    distress_dial_reduction: 2,
  },

  desktop: {
    enabled: true,
    require_confirmation: true,
    command_timeout_seconds: 30,
  },

  internet: {
    internet_enabled: true,
    searxng_url: "",
    web_cache_ttl_s: 3600,
  },

  setPersonality: (personality) => set({ personality }),
  setDesktop: (desktop) => set({ desktop }),
  setInternet: (internet) => set({ internet }),

  // -- Voice ----------------------------------------------------------------
  voiceStatus: {
    state: "idle",
    wake_word_enabled: false,
    hotkey_enabled: false,
    stt_model: "",
    tts_model: "",
    tts_voice: "",
  },

  setVoiceStatus: (voiceStatus) => set({ voiceStatus }),

  // -- Connection -----------------------------------------------------------
  isConnected: false,
  setConnected: (isConnected) => set({ isConnected }),
}));
