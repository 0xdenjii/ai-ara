"use client";

import { motion } from "framer-motion";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import type { MessageRole } from "@/types/api";

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface ChatMessageProps {
  role: MessageRole;
  content: string;
  isStreaming?: boolean;
  timestamp?: string;
  intent?: string;
  agent?: string;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatTimestamp(iso: string): string {
  try {
    return new Date(iso).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return "";
  }
}

// ---------------------------------------------------------------------------
// Typing dots indicator for empty streaming messages
// ---------------------------------------------------------------------------

function TypingIndicator() {
  return (
    <span className="inline-flex items-center gap-1" aria-label="Typing">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="size-1.5 rounded-full bg-muted-foreground/60 animate-pulse"
          style={{ animationDelay: `${i * 150}ms` }}
        />
      ))}
    </span>
  );
}

// ---------------------------------------------------------------------------
// ARA Avatar
// ---------------------------------------------------------------------------

function AraAvatar() {
  return (
    <div className="flex size-8 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-indigo-500 to-violet-600 text-xs font-bold text-white shadow-lg shadow-indigo-500/20">
      A
    </div>
  );
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function ChatMessage({
  role,
  content,
  isStreaming = false,
  timestamp,
  intent,
  agent,
}: ChatMessageProps) {
  const isUser = role === "user";
  const isSystem = role === "system";
  const isEmpty = !content || content.trim().length === 0;

  // System messages render as a centered subtle note
  if (isSystem) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.25 }}
        className="flex justify-center py-2"
      >
        <span className="rounded-md bg-muted/50 px-3 py-1.5 text-xs text-muted-foreground">
          {content}
        </span>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
      className={cn(
        "group flex gap-3 px-4 py-2",
        isUser ? "flex-row-reverse" : "flex-row"
      )}
    >
      {/* Avatar — assistant only */}
      {!isUser && <AraAvatar />}

      {/* Bubble */}
      <div
        className={cn(
          "relative max-w-[75%] rounded-2xl px-4 py-3 text-sm leading-relaxed",
          isUser
            ? "bg-gradient-to-br from-indigo-600 to-violet-600 text-white shadow-lg shadow-indigo-500/10"
            : "bg-card border border-border text-card-foreground shadow-sm"
        )}
      >
        {/* Content */}
        {isEmpty && isStreaming ? (
          <TypingIndicator />
        ) : isUser ? (
          <p className="whitespace-pre-wrap break-words">{content}</p>
        ) : (
          <div className="markdown-prose max-w-none text-card-foreground">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              rehypePlugins={[rehypeHighlight]}
            >
              {content}
            </ReactMarkdown>
            {/* Streaming cursor */}
            {isStreaming && (
              <span className="streaming-cursor ml-0.5 inline-block text-indigo-400 font-mono">
                &#9612;
              </span>
            )}
          </div>
        )}

        {/* Meta: intent / agent badges */}
        {(intent || agent) && (
          <div
            className={cn(
              "mt-2 flex flex-wrap gap-1.5",
              isUser ? "justify-end" : "justify-start"
            )}
          >
            {intent && (
              <Badge
                variant="secondary"
                className="text-[10px] px-1.5 py-0 h-4 bg-white/10 text-muted-foreground"
              >
                {intent}
              </Badge>
            )}
            {agent && (
              <Badge
                variant="secondary"
                className="text-[10px] px-1.5 py-0 h-4 bg-indigo-500/15 text-indigo-400"
              >
                {agent}
              </Badge>
            )}
          </div>
        )}

        {/* Timestamp */}
        {timestamp && (
          <div
            className={cn(
              "mt-1.5 text-[10px] opacity-0 transition-opacity group-hover:opacity-100",
              isUser
                ? "text-white/50 text-right"
                : "text-muted-foreground text-left"
            )}
          >
            {formatTimestamp(timestamp)}
          </div>
        )}
      </div>
    </motion.div>
  );
}
