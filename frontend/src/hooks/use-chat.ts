"use client";

import { useRef, useCallback } from "react";
import { useAppStore } from "@/lib/store";
import type { MessageItem, StreamChunk } from "@/types/api";

const API_BASE = "http://127.0.0.1:8000";

// ---------------------------------------------------------------------------
// Return type
// ---------------------------------------------------------------------------

interface UseChatReturn {
  messages: MessageItem[];
  isStreaming: boolean;
  sendMessage: (content: string) => Promise<void>;
  stopStreaming: () => void;
  clearChat: () => void;
}

// ---------------------------------------------------------------------------
// SSE stream helper
// ---------------------------------------------------------------------------

async function* streamChat(
  message: string,
  conversationId: string,
  signal: AbortSignal,
): AsyncGenerator<StreamChunk> {
  const response = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      conversation_id: conversationId || undefined,
      stream: true,
    }),
    signal,
  });

  if (!response.ok) {
    throw new Error(`Chat request failed: ${response.status}`);
  }

  const reader = response.body?.getReader();
  if (!reader) throw new Error("No readable stream in response");

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // SSE frames are separated by double newlines.
    const parts = buffer.split("\n\n");
    buffer = parts.pop() ?? "";

    for (const part of parts) {
      for (const line of part.split("\n")) {
        if (line.startsWith("data: ")) {
          const json = line.slice(6).trim();
          if (json === "[DONE]") return;
          try {
            yield JSON.parse(json) as StreamChunk;
          } catch {
            // skip malformed chunks
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useChat(): UseChatReturn {
  const messages = useAppStore((s) => s.messages);
  const isStreaming = useAppStore((s) => s.isStreaming);
  const currentConversationId = useAppStore((s) => s.currentConversationId);
  const addMessage = useAppStore((s) => s.addMessage);
  const setMessages = useAppStore((s) => s.setMessages);
  const setStreaming = useAppStore((s) => s.setStreaming);
  const setConversationId = useAppStore((s) => s.setConversationId);

  const abortRef = useRef<AbortController | null>(null);

  const sendMessage = useCallback(
    async (content: string) => {
      if (!content.trim()) return;

      // 1. Add user message to store.
      const userMessage: MessageItem = {
        role: "user",
        content,
        timestamp: new Date().toISOString(),
      };
      addMessage(userMessage);

      // 2. Prepare streaming state.
      setStreaming(true);
      const controller = new AbortController();
      abortRef.current = controller;

      // Placeholder assistant message that we'll update token-by-token.
      const assistantMessage: MessageItem = {
        role: "assistant",
        content: "",
        timestamp: new Date().toISOString(),
      };

      // Snapshot current messages + the user message + empty assistant stub.
      const baseMessages = [
        ...useAppStore.getState().messages,
      ];
      // `addMessage` already appended the user message above, so baseMessages
      // already contains it. Append the assistant stub.
      const withAssistant = [...baseMessages, assistantMessage];
      setMessages(withAssistant);

      try {
        let accumulated = "";

        for await (const chunk of streamChat(
          content,
          currentConversationId,
          controller.signal,
        )) {
          if (chunk.type === "token") {
            accumulated += chunk.content;

            // Update the last (assistant) message in place.
            const updated: MessageItem = {
              ...assistantMessage,
              content: accumulated,
            };
            setMessages([...baseMessages, updated]);
          } else if (chunk.type === "done") {
            // The server may include a conversation_id in metadata.
            const convId =
              (chunk.metadata?.conversation_id as string) ?? "";
            if (convId) setConversationId(convId);
          } else if (chunk.type === "error") {
            const errorMsg: MessageItem = {
              role: "assistant",
              content: `Error: ${chunk.content}`,
              timestamp: new Date().toISOString(),
            };
            setMessages([...baseMessages, errorMsg]);
          }
        }
      } catch (err: unknown) {
        // Ignore abort errors — those are user-initiated.
        if (err instanceof DOMException && err.name === "AbortError") {
          return;
        }

        const errorText =
          err instanceof Error ? err.message : "Unknown error";
        const errorMsg: MessageItem = {
          role: "assistant",
          content: `Error: ${errorText}`,
          timestamp: new Date().toISOString(),
        };
        setMessages([...baseMessages, errorMsg]);
      } finally {
        setStreaming(false);
        abortRef.current = null;
      }
    },
    [
      addMessage,
      setMessages,
      setStreaming,
      setConversationId,
      currentConversationId,
    ],
  );

  const stopStreaming = useCallback(() => {
    abortRef.current?.abort();
    setStreaming(false);
  }, [setStreaming]);

  const clearChat = useAppStore((s) => s.clearChat);

  return { messages, isStreaming, sendMessage, stopStreaming, clearChat };
}
