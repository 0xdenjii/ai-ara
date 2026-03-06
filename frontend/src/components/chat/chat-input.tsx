"use client";

import { useRef, useState, useCallback, type KeyboardEvent } from "react";
import { ArrowUp, Square, Paperclip } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface ChatInputProps {
  onSend: (message: string) => void;
  isStreaming: boolean;
  onStop: () => void;
  onImageUpload?: (file: File) => void;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function ChatInput({
  onSend,
  isStreaming,
  onStop,
  onImageUpload,
}: ChatInputProps) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const canSend = value.trim().length > 0 && !isStreaming;

  // ---- Submit handler ------------------------------------------------------

  const handleSend = useCallback(() => {
    const trimmed = value.trim();
    if (!trimmed || isStreaming) return;
    onSend(trimmed);
    setValue("");
    // Reset textarea height after send
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  }, [value, isStreaming, onSend]);

  // ---- Keyboard handler (Enter to send, Shift+Enter for newline) ----------

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend]
  );

  // ---- Auto-resize textarea ------------------------------------------------

  const handleInput = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    // Max 6 rows (~144px at 24px line-height)
    const maxHeight = 144;
    el.style.height = `${Math.min(el.scrollHeight, maxHeight)}px`;
  }, []);

  // ---- File upload ---------------------------------------------------------

  const handleFileClick = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file && onImageUpload) {
        onImageUpload(file);
      }
      // Reset so the same file can be re-selected
      e.target.value = "";
    },
    [onImageUpload]
  );

  return (
    <div className="sticky bottom-0 z-10 w-full px-4 pb-4 pt-2">
      {/* Glass-morphism container */}
      <div
        className={cn(
          "mx-auto flex max-w-3xl items-end gap-2 rounded-2xl",
          "border border-border/50 bg-background/70 backdrop-blur-xl",
          "px-3 py-2 shadow-lg shadow-black/10",
          "transition-colors focus-within:border-indigo-500/50 focus-within:shadow-indigo-500/5"
        )}
      >
        {/* Image upload button */}
        {onImageUpload && (
          <>
            <Button
              type="button"
              variant="ghost"
              size="icon-sm"
              className="shrink-0 text-muted-foreground hover:text-foreground"
              onClick={handleFileClick}
              aria-label="Attach image"
            >
              <Paperclip className="size-4" />
            </Button>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={handleFileChange}
            />
          </>
        )}

        {/* Textarea */}
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => {
            setValue(e.target.value);
            handleInput();
          }}
          onKeyDown={handleKeyDown}
          placeholder="Message ARA..."
          rows={1}
          className={cn(
            "flex-1 resize-none bg-transparent text-sm leading-6 text-foreground",
            "placeholder:text-muted-foreground/60",
            "outline-none",
            "max-h-36 scrollbar-thin"
          )}
        />

        {/* Send / Stop button */}
        {isStreaming ? (
          <Button
            type="button"
            size="icon-sm"
            variant="destructive"
            className="shrink-0 rounded-lg"
            onClick={onStop}
            aria-label="Stop generating"
          >
            <Square className="size-3.5" />
          </Button>
        ) : (
          <Button
            type="button"
            size="icon-sm"
            className={cn(
              "shrink-0 rounded-lg transition-all",
              canSend
                ? "bg-indigo-600 text-white hover:bg-indigo-500 shadow-md shadow-indigo-500/20"
                : "bg-muted text-muted-foreground"
            )}
            disabled={!canSend}
            onClick={handleSend}
            aria-label="Send message"
          >
            <ArrowUp className="size-4" />
          </Button>
        )}
      </div>

      {/* Hint text */}
      <p className="mt-1.5 text-center text-[10px] text-muted-foreground/50">
        ARA can make mistakes. Double-check important info.
      </p>
    </div>
  );
}
