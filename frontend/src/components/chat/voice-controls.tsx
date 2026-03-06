"use client";

import { useCallback, useEffect, useState } from "react";
import { Mic, MicOff } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { controlVoice, fetchVoiceStatus } from "@/lib/api";
import { cn } from "@/lib/utils";
import type { VoiceStatusResponse } from "@/types/api";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type VoiceState = "idle" | "listening" | "processing" | "speaking";

// ---------------------------------------------------------------------------
// Sound wave bars animation (CSS keyframes injected via style tag)
// ---------------------------------------------------------------------------

const soundWaveKeyframes = `
@keyframes ara-voice-bar {
  0%, 100% { transform: scaleY(0.3); }
  50% { transform: scaleY(1); }
}
`;

function SoundWaveBars() {
  const bars = [
    { delay: "0ms", height: "h-3" },
    { delay: "120ms", height: "h-4" },
    { delay: "60ms", height: "h-5" },
    { delay: "180ms", height: "h-4" },
    { delay: "90ms", height: "h-3" },
  ];

  return (
    <>
      <style>{soundWaveKeyframes}</style>
      <div className="flex items-center justify-center gap-0.5">
        {bars.map((bar, i) => (
          <span
            key={i}
            className={cn(
              "w-[3px] origin-center rounded-full bg-indigo-400",
              bar.height,
            )}
            style={{
              animation: "ara-voice-bar 0.6s ease-in-out infinite",
              animationDelay: bar.delay,
            }}
          />
        ))}
      </div>
    </>
  );
}

// ---------------------------------------------------------------------------
// Status label
// ---------------------------------------------------------------------------

const STATUS_LABELS: Record<VoiceState, string> = {
  idle: "Ready",
  listening: "Listening...",
  processing: "Processing...",
  speaking: "Speaking...",
};

const STATUS_COLORS: Record<VoiceState, string> = {
  idle: "text-muted-foreground",
  listening: "text-indigo-400",
  processing: "text-amber-400",
  speaking: "text-emerald-400",
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function VoiceControls() {
  const [voiceState, setVoiceState] = useState<VoiceState>("idle");
  const [pressing, setPressing] = useState(false);
  const [wakeWordEnabled, setWakeWordEnabled] = useState(false);
  const [hotkeyEnabled, setHotkeyEnabled] = useState(false);

  // Fetch initial voice status
  useEffect(() => {
    fetchVoiceStatus()
      .then((status: VoiceStatusResponse) => {
        setVoiceState((status.state as VoiceState) || "idle");
        setWakeWordEnabled(status.wake_word_enabled);
        setHotkeyEnabled(status.hotkey_enabled);
      })
      .catch(() => {
        // Voice service may not be running
      });
  }, []);

  // ---- Push-to-talk handlers -----------------------------------------------

  const handlePointerDown = useCallback(() => {
    setPressing(true);
    setVoiceState("listening");
    controlVoice({ action: "start" }).catch(() => {
      setVoiceState("idle");
      setPressing(false);
    });
  }, []);

  const handlePointerUp = useCallback(() => {
    setPressing(false);
    if (voiceState === "listening") {
      setVoiceState("processing");
      controlVoice({ action: "stop" })
        .then(() => {
          // Simulate returning to idle after processing
          setTimeout(() => setVoiceState("idle"), 1500);
        })
        .catch(() => {
          setVoiceState("idle");
        });
    }
  }, [voiceState]);

  // ---- Toggle handlers -----------------------------------------------------

  const toggleWakeWord = useCallback(() => {
    controlVoice({ action: "toggle_wake_word" })
      .then((res) => {
        setWakeWordEnabled(!wakeWordEnabled);
        if (res.state) setVoiceState(res.state as VoiceState);
      })
      .catch(() => {});
  }, [wakeWordEnabled]);

  const toggleHotkey = useCallback(() => {
    controlVoice({ action: "toggle_hotkey" })
      .then((res) => {
        setHotkeyEnabled(!hotkeyEnabled);
        if (res.state) setVoiceState(res.state as VoiceState);
      })
      .catch(() => {});
  }, [hotkeyEnabled]);

  const isActive = voiceState === "listening" || voiceState === "speaking";

  return (
    <div className="flex items-center gap-3">
      {/* Push-to-talk button */}
      <div className="relative flex flex-col items-center gap-1.5">
        {/* Pulsing ring when pressing */}
        {pressing && (
          <div className="absolute inset-0 -m-1 rounded-full animate-ping bg-indigo-500/30" />
        )}

        <button
          type="button"
          onPointerDown={handlePointerDown}
          onPointerUp={handlePointerUp}
          onPointerLeave={() => {
            if (pressing) handlePointerUp();
          }}
          className={cn(
            "relative z-10 flex size-12 items-center justify-center rounded-full",
            "border-2 transition-all duration-200 select-none",
            "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
            pressing
              ? "border-indigo-500 bg-indigo-600 shadow-lg shadow-indigo-500/30 scale-95"
              : "border-border/50 bg-card hover:border-border hover:bg-card/80",
          )}
          aria-label="Push to talk"
        >
          {isActive ? (
            <SoundWaveBars />
          ) : pressing ? (
            <Mic className="size-5 text-white" />
          ) : (
            <MicOff className="size-5 text-muted-foreground" />
          )}
        </button>

        {/* Voice state label */}
        <span
          className={cn(
            "text-[10px] font-medium transition-colors",
            STATUS_COLORS[voiceState],
          )}
        >
          {STATUS_LABELS[voiceState]}
        </span>
      </div>

      {/* Toggles */}
      <div className="flex flex-col gap-1">
        <Button
          variant={wakeWordEnabled ? "default" : "outline"}
          size="xs"
          onClick={toggleWakeWord}
          className={cn(
            "text-[10px] h-5 px-2",
            wakeWordEnabled
              ? "bg-indigo-600 text-white hover:bg-indigo-500"
              : "text-muted-foreground",
          )}
        >
          Hey ARA
        </Button>
        <Button
          variant={hotkeyEnabled ? "default" : "outline"}
          size="xs"
          onClick={toggleHotkey}
          className={cn(
            "text-[10px] h-5 px-2",
            hotkeyEnabled
              ? "bg-indigo-600 text-white hover:bg-indigo-500"
              : "text-muted-foreground",
          )}
        >
          Ctrl+Space
        </Button>
      </div>

      {/* Status badge (visible on wider layouts) */}
      {voiceState !== "idle" && (
        <Badge
          variant="outline"
          className={cn(
            "hidden sm:inline-flex border text-[10px] gap-1",
            voiceState === "listening" && "border-indigo-500/30 text-indigo-400",
            voiceState === "processing" && "border-amber-500/30 text-amber-400",
            voiceState === "speaking" && "border-emerald-500/30 text-emerald-400",
          )}
        >
          <span className="relative flex size-1.5">
            <span className="absolute inline-flex size-full animate-ping rounded-full bg-current opacity-75" />
            <span className="relative inline-flex size-1.5 rounded-full bg-current" />
          </span>
          {STATUS_LABELS[voiceState]}
        </Badge>
      )}
    </div>
  );
}
