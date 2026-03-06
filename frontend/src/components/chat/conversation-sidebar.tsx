"use client";

import { useEffect, useState, useMemo, useCallback } from "react";
import { Plus, Search, MessageSquare } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { fetchConversations, fetchConversation } from "@/lib/api";
import { useAppStore } from "@/lib/store";
import type { ConversationItem } from "@/types/api";

// ---------------------------------------------------------------------------
// Relative time helper
// ---------------------------------------------------------------------------

function relativeTime(iso: string): string {
  try {
    const date = new Date(iso);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffSec = Math.floor(diffMs / 1000);
    const diffMin = Math.floor(diffSec / 60);
    const diffHr = Math.floor(diffMin / 60);
    const diffDay = Math.floor(diffHr / 24);

    if (diffSec < 60) return "Just now";
    if (diffMin < 60) return `${diffMin}m ago`;
    if (diffHr < 24) return `${diffHr}h ago`;
    if (diffDay === 1) return "Yesterday";
    if (diffDay < 7) return `${diffDay}d ago`;
    return date.toLocaleDateString([], { month: "short", day: "numeric" });
  } catch {
    return "";
  }
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function ConversationSidebar() {
  const conversations = useAppStore((s) => s.conversations);
  const setConversations = useAppStore((s) => s.setConversations);
  const currentConversationId = useAppStore((s) => s.currentConversationId);
  const setConversationId = useAppStore((s) => s.setConversationId);
  const setMessages = useAppStore((s) => s.setMessages);
  const clearChat = useAppStore((s) => s.clearChat);

  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(false);

  // ---- Load conversations on mount -----------------------------------------

  useEffect(() => {
    let cancelled = false;

    async function load() {
      setLoading(true);
      try {
        const data = await fetchConversations(50);
        if (!cancelled) {
          setConversations(data.conversations);
        }
      } catch {
        // Silently fail — user can retry or the sidebar just shows empty.
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, [setConversations]);

  // ---- Filtered list -------------------------------------------------------

  const filtered = useMemo(() => {
    if (!search.trim()) return conversations;
    const q = search.toLowerCase();
    return conversations.filter((c) =>
      c.summary.toLowerCase().includes(q)
    );
  }, [conversations, search]);

  // ---- Load conversation handler -------------------------------------------

  const handleSelect = useCallback(
    async (item: ConversationItem) => {
      if (item.conversation_id === currentConversationId) return;
      try {
        const detail = await fetchConversation(item.conversation_id);
        setConversationId(detail.conversation_id);
        setMessages(detail.turns);
      } catch {
        // Could show a toast here, but keep it simple
      }
    },
    [currentConversationId, setConversationId, setMessages]
  );

  // ---- New chat handler ----------------------------------------------------

  const handleNewChat = useCallback(() => {
    clearChat();
  }, [clearChat]);

  return (
    <div className="flex h-full flex-col bg-card/50">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-border/50 px-3 py-3">
        <h2 className="text-sm font-semibold text-foreground">Conversations</h2>
        <Button
          variant="ghost"
          size="icon-sm"
          onClick={handleNewChat}
          aria-label="New chat"
          className="text-muted-foreground hover:text-foreground"
        >
          <Plus className="size-4" />
        </Button>
      </div>

      {/* Search */}
      <div className="px-3 py-2">
        <div className="relative">
          <Search className="absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground/50" />
          <Input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search..."
            className="h-8 pl-8 text-xs bg-background/50 border-border/50"
          />
        </div>
      </div>

      {/* List */}
      <ScrollArea className="flex-1">
        <div className="space-y-0.5 px-2 py-1">
          {loading && conversations.length === 0 && (
            <div className="flex items-center justify-center py-8">
              <span className="text-xs text-muted-foreground animate-pulse">
                Loading...
              </span>
            </div>
          )}

          {!loading && filtered.length === 0 && (
            <div className="flex flex-col items-center justify-center gap-2 py-8 text-muted-foreground">
              <MessageSquare className="size-5 opacity-40" />
              <span className="text-xs">
                {search ? "No matches found" : "No conversations yet"}
              </span>
            </div>
          )}

          {filtered.map((item) => {
            const isActive = item.conversation_id === currentConversationId;
            return (
              <button
                key={item.conversation_id}
                onClick={() => handleSelect(item)}
                className={cn(
                  "flex w-full flex-col gap-0.5 rounded-lg px-3 py-2.5 text-left transition-colors",
                  isActive
                    ? "bg-indigo-500/10 border border-indigo-500/20 text-foreground"
                    : "border border-transparent text-muted-foreground hover:bg-accent/50 hover:text-foreground"
                )}
              >
                {/* Summary */}
                <span className="truncate text-xs font-medium leading-snug">
                  {item.summary || "Untitled conversation"}
                </span>

                {/* Meta row */}
                <div className="flex items-center gap-2 text-[10px] opacity-70">
                  <span>{item.turn_count} turns</span>
                  {item.started_at && (
                    <>
                      <span className="text-border">&middot;</span>
                      <span>{relativeTime(item.started_at)}</span>
                    </>
                  )}
                </div>
              </button>
            );
          })}
        </div>
      </ScrollArea>
    </div>
  );
}
