"use client";

import { useEffect, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  PanelLeftClose,
  PanelLeftOpen,
  Sparkles,
  BookOpen,
  Target,
  Laugh,
  CloudSun,
} from "lucide-react";
import { toast } from "sonner";

import { useChat } from "@/hooks/use-chat";
import { useAppStore } from "@/lib/store";
import { cn } from "@/lib/utils";
import { fetchHealth } from "@/lib/api";

import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { ChatMessage } from "@/components/chat/chat-message";
import { ChatInput } from "@/components/chat/chat-input";
import { ConversationSidebar } from "@/components/chat/conversation-sidebar";

// ---------------------------------------------------------------------------
// Quick-action suggestions for the empty state
// ---------------------------------------------------------------------------

const suggestions = [
  {
    label: "What did I write about last week?",
    icon: BookOpen,
  },
  {
    label: "Summarize my goals",
    icon: Target,
  },
  {
    label: "Tell me a joke",
    icon: Laugh,
  },
  {
    label: "What's the weather?",
    icon: CloudSun,
  },
];

// ---------------------------------------------------------------------------
// Page Component
// ---------------------------------------------------------------------------

export default function ChatPage() {
  const { messages, isStreaming, sendMessage, stopStreaming } = useChat();

  const sidebarOpen = useAppStore((s) => s.sidebarOpen);
  const toggleSidebar = useAppStore((s) => s.toggleSidebar);
  const isConnected = useAppStore((s) => s.isConnected);
  const setConnected = useAppStore((s) => s.setConnected);

  const scrollRef = useRef<HTMLDivElement>(null);
  const prevConnected = useRef(isConnected);

  // ---- Auto-scroll on new messages -----------------------------------------

  useEffect(() => {
    const el = scrollRef.current;
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "end" });
    }
  }, [messages]);

  // ---- Connection health check ---------------------------------------------

  useEffect(() => {
    let cancelled = false;

    async function check() {
      try {
        await fetchHealth();
        if (!cancelled) setConnected(true);
      } catch {
        if (!cancelled) setConnected(false);
      }
    }

    check();
    const interval = setInterval(check, 15_000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [setConnected]);

  // ---- Connection status toasts --------------------------------------------

  useEffect(() => {
    if (prevConnected.current !== isConnected) {
      if (isConnected) {
        toast.success("Connected to ARA backend");
      } else {
        toast.error("Disconnected from ARA backend");
      }
      prevConnected.current = isConnected;
    }
  }, [isConnected]);

  // ---- Send handler (also used by quick-actions) ---------------------------

  const handleSend = useCallback(
    (text: string) => {
      sendMessage(text);
    },
    [sendMessage]
  );

  const isEmpty = messages.length === 0;

  return (
    <div className="flex h-full">
      {/* ---- Conversation Sidebar ---- */}
      <AnimatePresence mode="wait">
        {sidebarOpen && (
          <motion.aside
            key="sidebar"
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: 280, opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            transition={{ duration: 0.2, ease: "easeInOut" }}
            className="hidden shrink-0 overflow-hidden border-r border-border/50 md:block"
          >
            <div className="h-full w-[280px]">
              <ConversationSidebar />
            </div>
          </motion.aside>
        )}
      </AnimatePresence>

      {/* ---- Main Chat Area ---- */}
      <div className="relative flex flex-1 flex-col overflow-hidden">
        {/* Top bar with sidebar toggle */}
        <div className="flex h-12 items-center gap-2 border-b border-border/30 px-3">
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={toggleSidebar}
            aria-label={sidebarOpen ? "Close sidebar" : "Open sidebar"}
            className="text-muted-foreground hover:text-foreground"
          >
            {sidebarOpen ? (
              <PanelLeftClose className="size-4" />
            ) : (
              <PanelLeftOpen className="size-4" />
            )}
          </Button>

          {/* Connection dot */}
          <div className="ml-auto flex items-center gap-1.5 text-[10px] text-muted-foreground">
            <span
              className={cn(
                "size-1.5 rounded-full",
                isConnected ? "bg-emerald-500" : "bg-red-500"
              )}
            />
            {isConnected ? "Online" : "Offline"}
          </div>
        </div>

        {/* Messages area */}
        <ScrollArea className="flex-1">
          <div className="mx-auto w-full max-w-3xl">
            {isEmpty ? (
              /* ---- Empty state ---- */
              <div className="flex h-[calc(100vh-12rem)] flex-col items-center justify-center gap-6 px-4">
                {/* ARA logo / circle */}
                <motion.div
                  initial={{ scale: 0.8, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  transition={{ duration: 0.5, ease: "easeOut" }}
                  className="flex size-20 items-center justify-center rounded-full bg-gradient-to-br from-indigo-500 to-violet-600 shadow-2xl shadow-indigo-500/25"
                >
                  <Sparkles className="size-8 text-white" />
                </motion.div>

                <motion.div
                  initial={{ y: 10, opacity: 0 }}
                  animate={{ y: 0, opacity: 1 }}
                  transition={{ delay: 0.15, duration: 0.4 }}
                  className="text-center"
                >
                  <h1 className="text-2xl font-semibold tracking-tight text-foreground">
                    ARA
                  </h1>
                  <p className="mt-1 text-sm text-muted-foreground">
                    How can I help you today?
                  </p>
                </motion.div>

                {/* Quick-action suggestion pills */}
                <motion.div
                  initial={{ y: 10, opacity: 0 }}
                  animate={{ y: 0, opacity: 1 }}
                  transition={{ delay: 0.3, duration: 0.4 }}
                  className="grid grid-cols-2 gap-2 sm:flex sm:flex-wrap sm:justify-center"
                >
                  {suggestions.map((s) => (
                    <button
                      key={s.label}
                      onClick={() => handleSend(s.label)}
                      className={cn(
                        "flex items-center gap-2 rounded-xl border border-border/50",
                        "bg-card/50 px-4 py-2.5 text-xs text-muted-foreground",
                        "transition-all hover:border-indigo-500/30 hover:bg-indigo-500/5 hover:text-foreground",
                        "hover:shadow-sm"
                      )}
                    >
                      <s.icon className="size-3.5 shrink-0 text-indigo-400" />
                      <span className="truncate">{s.label}</span>
                    </button>
                  ))}
                </motion.div>
              </div>
            ) : (
              /* ---- Message list ---- */
              <div className="space-y-1 py-4">
                {messages.map((msg, i) => {
                  const isLast = i === messages.length - 1;
                  return (
                    <ChatMessage
                      key={`${msg.role}-${i}`}
                      role={msg.role}
                      content={msg.content}
                      timestamp={msg.timestamp}
                      isStreaming={isLast && isStreaming && msg.role === "assistant"}
                    />
                  );
                })}
                {/* Scroll anchor */}
                <div ref={scrollRef} />
              </div>
            )}
          </div>
        </ScrollArea>

        {/* Chat input — pinned to bottom */}
        <ChatInput
          onSend={handleSend}
          isStreaming={isStreaming}
          onStop={stopStreaming}
        />
      </div>
    </div>
  );
}
