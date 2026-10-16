"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import {
  Settings,
  Smile,
  Mic,
  Monitor,
  Wifi,
  WifiOff,
  Info,
  Loader2,
  Play,
  Square,
  Shield,
  Clock,
  Globe,
  Heart,
  Flame,
  Zap,
  Skull,
  Save,
} from "lucide-react";
import { toast } from "sonner";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Separator } from "@/components/ui/separator";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
import {
  fetchSettings,
  updatePersonality,
  updateDesktop,
  updateInternet,
  fetchVoiceStatus,
  controlVoice,
  fetchHealth,
} from "@/lib/api";
import type {
  SettingsResponse,
  VoiceStatusResponse,
  HealthResponse,
} from "@/types/api";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const HUMOR_LABELS: Record<number, { label: string; icon: React.ReactNode }> = {
  1: { label: "Supportive Friend", icon: <Heart className="size-4 text-blue-400" /> },
  2: { label: "Supportive Friend", icon: <Heart className="size-4 text-blue-400" /> },
  3: { label: "Supportive Friend", icon: <Heart className="size-4 text-blue-400" /> },
  4: { label: "Warm Sarcastic", icon: <Smile className="size-4 text-purple-400" /> },
  5: { label: "Warm Sarcastic", icon: <Smile className="size-4 text-purple-400" /> },
  6: { label: "Witty Irreverent", icon: <Flame className="size-4 text-orange-400" /> },
  7: { label: "Witty Irreverent", icon: <Flame className="size-4 text-orange-400" /> },
  8: { label: "Full Grok Mode", icon: <Zap className="size-4 text-red-400" /> },
  9: { label: "Full Grok Mode", icon: <Zap className="size-4 text-red-400" /> },
  10: { label: "Unfiltered Chaos", icon: <Skull className="size-4 text-red-500" /> },
};

function formatUptime(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  if (h > 0) return `${h}h ${m}m ${s}s`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function SettingsPage() {
  // Settings state
  const [settings, setSettings] = useState<SettingsResponse | null>(null);
  const [voice, setVoice] = useState<VoiceStatusResponse | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Personality local state (for live editing before save)
  const [humorDial, setHumorDial] = useState(5);
  const [autoDialDown, setAutoDialDown] = useState(true);
  const [distressReduction, setDistressReduction] = useState(2);
  const [personalitySaving, setPersonalitySaving] = useState(false);
  const [personalityDirty, setPersonalityDirty] = useState(false);

  // Desktop local state
  const [desktopEnabled, setDesktopEnabled] = useState(false);
  const [desktopConfirmation, setDesktopConfirmation] = useState(true);

  // Internet local state
  const [internetEnabled, setInternetEnabled] = useState(false);

  // Voice control state
  const [voiceLoading, setVoiceLoading] = useState(false);

  // Debounce timer refs
  const desktopTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const internetTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // ---- Load data --------------------------------------------------------
  const loadData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const [s, v, h] = await Promise.allSettled([
        fetchSettings(),
        fetchVoiceStatus(),
        fetchHealth(),
      ]);

      if (s.status === "fulfilled") {
        const st = s.value;
        setSettings(st);
        setHumorDial(st.personality.humor_dial);
        setAutoDialDown(st.personality.auto_dial_down_on_distress);
        setDistressReduction(st.personality.distress_dial_reduction);
        setDesktopEnabled(st.desktop.enabled);
        setDesktopConfirmation(st.desktop.require_confirmation);
        setInternetEnabled(st.internet.internet_enabled);
      }
      if (v.status === "fulfilled") setVoice(v.value);
      if (h.status === "fulfilled") setHealth(h.value);

      // If all failed, show error
      if (
        s.status === "rejected" &&
        v.status === "rejected" &&
        h.status === "rejected"
      ) {
        setError("Failed to connect to ARA backend");
      }
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  // ---- Personality save -------------------------------------------------
  const savePersonality = async () => {
    try {
      setPersonalitySaving(true);
      const result = await updatePersonality({
        humor_dial: humorDial,
        auto_dial_down_on_distress: autoDialDown,
        distress_dial_reduction: distressReduction,
      });
      setSettings((prev) =>
        prev ? { ...prev, personality: result } : prev,
      );
      setPersonalityDirty(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save personality");
    } finally {
      setPersonalitySaving(false);
    }
  };

  // ---- Desktop auto-save (debounced) ------------------------------------
  const saveDesktop = useCallback(
    (enabled: boolean, confirm: boolean) => {
      if (desktopTimer.current) clearTimeout(desktopTimer.current);
      desktopTimer.current = setTimeout(async () => {
        try {
          const result = await updateDesktop({
            enabled,
            require_confirmation: confirm,
          });
          setSettings((prev) =>
            prev ? { ...prev, desktop: result } : prev,
          );
        } catch {
          // silently fail on debounced save
        }
      }, 600);
    },
    [],
  );

  // ---- Internet auto-save (debounced) -----------------------------------
  const saveInternet = useCallback(
    (enabled: boolean) => {
      if (internetTimer.current) clearTimeout(internetTimer.current);
      internetTimer.current = setTimeout(async () => {
        try {
          const result = await updateInternet({ internet_enabled: enabled });
          setSettings((prev) =>
            prev ? { ...prev, internet: result } : prev,
          );
        } catch (err) {
          toast.error(
            `Failed to save internet settings: ${err instanceof Error ? err.message : "Unknown error"}`,
          );
        }
      }, 600);
    },
    [],
  );

  // ---- Voice control ----------------------------------------------------
  const handleVoiceAction = async (
    action: "start" | "stop" | "toggle_wake_word" | "toggle_hotkey",
  ) => {
    try {
      setVoiceLoading(true);
      const result = await controlVoice({ action });
      if (result.success) {
        // Refresh voice status
        const v = await fetchVoiceStatus();
        setVoice(v);
      }
    } catch (err) {
      toast.error(
        `Voice action failed: ${err instanceof Error ? err.message : "Unknown error"}`,
      );
    } finally {
      setVoiceLoading(false);
    }
  };

  // ---- Humor dial gradient color ----------------------------------------
  function humorColor(val: number): string {
    if (val <= 3) return "text-blue-400";
    if (val <= 5) return "text-purple-400";
    if (val <= 7) return "text-orange-400";
    if (val <= 9) return "text-red-400";
    return "text-red-500";
  }

  return (
    <div className="min-h-screen bg-background p-6 md:p-8">
      <div className="mx-auto max-w-4xl space-y-8">
        {/* ---- Header -------------------------------------------------- */}
        <div className="flex items-center gap-3">
          <div className="flex size-10 items-center justify-center rounded-lg bg-violet-500/10 text-violet-400">
            <Settings className="size-5" />
          </div>
          <div>
            <h1 className="text-2xl font-bold tracking-tight">Settings</h1>
            <p className="text-sm text-muted-foreground">
              Configure ARA&apos;s personality, voice, desktop automation, and
              more
            </p>
          </div>
        </div>

        {error && (
          <Card className="border-red-500/30 bg-red-500/5">
            <CardContent className="py-4 text-sm text-red-400">
              {error}
            </CardContent>
          </Card>
        )}

        {loading ? (
          <Card>
            <CardContent className="flex items-center justify-center py-16">
              <Loader2 className="mr-2 size-5 animate-spin text-muted-foreground" />
              <span className="text-muted-foreground">
                Loading settings...
              </span>
            </CardContent>
          </Card>
        ) : (
          <Tabs defaultValue="personality" className="space-y-6">
            <TabsList className="w-full justify-start">
              <TabsTrigger value="personality">
                <Smile className="mr-1.5 size-4" />
                Personality
              </TabsTrigger>
              <TabsTrigger value="voice">
                <Mic className="mr-1.5 size-4" />
                Voice
              </TabsTrigger>
              <TabsTrigger value="desktop">
                <Monitor className="mr-1.5 size-4" />
                Desktop
              </TabsTrigger>
              <TabsTrigger value="internet">
                <Globe className="mr-1.5 size-4" />
                Internet
              </TabsTrigger>
              <TabsTrigger value="about">
                <Info className="mr-1.5 size-4" />
                About
              </TabsTrigger>
            </TabsList>

            {/* ============================================================ */}
            {/* PERSONALITY TAB                                               */}
            {/* ============================================================ */}
            <TabsContent value="personality" className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-base">
                    <Smile className="size-4 text-violet-400" />
                    Humor Dial
                  </CardTitle>
                  <CardDescription>
                    Control ARA&apos;s personality from supportive to unhinged
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  {/* Current level display */}
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      {HUMOR_LABELS[humorDial]?.icon}
                      <span className={`font-medium ${humorColor(humorDial)}`}>
                        {HUMOR_LABELS[humorDial]?.label ?? "Unknown"}
                      </span>
                    </div>
                    <Badge variant="outline" className={humorColor(humorDial)}>
                      Level {humorDial}
                    </Badge>
                  </div>

                  {/* Slider */}
                  <div className="space-y-2">
                    <div className="relative">
                      {/* Gradient track underlay */}
                      <div className="absolute inset-x-0 top-1/2 h-1.5 -translate-y-1/2 rounded-full bg-gradient-to-r from-blue-500 via-purple-500 to-red-500 opacity-30" />
                      <Slider
                        min={1}
                        max={10}
                        step={1}
                        value={[humorDial]}
                        onValueChange={(val) => {
                          setHumorDial(val[0]);
                          setPersonalityDirty(true);
                        }}
                        className="relative [&_[data-slot=slider-range]]:bg-gradient-to-r [&_[data-slot=slider-range]]:from-blue-500 [&_[data-slot=slider-range]]:via-purple-500 [&_[data-slot=slider-range]]:to-red-500"
                      />
                    </div>
                    <div className="flex justify-between text-xs text-muted-foreground">
                      <span>Supportive</span>
                      <span>Unfiltered</span>
                    </div>
                  </div>

                  <Separator />

                  {/* Auto dial-down */}
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <p className="text-sm font-medium">
                        Auto Dial-Down on Distress
                      </p>
                      <p className="text-xs text-muted-foreground">
                        Automatically reduce humor when distress is detected
                      </p>
                    </div>
                    <Switch
                      checked={autoDialDown}
                      onCheckedChange={(v) => {
                        setAutoDialDown(v);
                        setPersonalityDirty(true);
                      }}
                    />
                  </div>

                  {/* Distress reduction amount */}
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <p className="text-sm font-medium">
                        Distress Dial Reduction
                      </p>
                      <p className="text-xs text-muted-foreground">
                        How many levels to reduce (0-5)
                      </p>
                    </div>
                    <Input
                      type="number"
                      min={0}
                      max={5}
                      value={distressReduction}
                      onChange={(e) => {
                        const val = Math.min(
                          5,
                          Math.max(0, parseInt(e.target.value) || 0),
                        );
                        setDistressReduction(val);
                        setPersonalityDirty(true);
                      }}
                      className="w-20 text-center"
                    />
                  </div>

                  {/* Save button */}
                  <div className="flex justify-end">
                    <Button
                      onClick={savePersonality}
                      disabled={personalitySaving || !personalityDirty}
                      className="bg-violet-600 text-white hover:bg-violet-700"
                    >
                      {personalitySaving ? (
                        <>
                          <Loader2 className="mr-2 size-4 animate-spin" />
                          Saving...
                        </>
                      ) : (
                        <>
                          <Save className="mr-2 size-4" />
                          Save Personality
                        </>
                      )}
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            {/* ============================================================ */}
            {/* VOICE TAB                                                     */}
            {/* ============================================================ */}
            <TabsContent value="voice" className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-base">
                    <Mic className="size-4 text-violet-400" />
                    Voice Pipeline
                  </CardTitle>
                  <CardDescription>
                    Manage voice recognition, wake word, and speech synthesis
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  {voice ? (
                    <>
                      {/* Status */}
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium">
                          Pipeline State
                        </span>
                        <Badge
                          variant={
                            voice.state === "running" ? "default" : "secondary"
                          }
                          className={
                            voice.state === "running"
                              ? "bg-emerald-500/15 text-emerald-400"
                              : ""
                          }
                        >
                          {voice.state}
                        </Badge>
                      </div>

                      {/* Models info */}
                      <div className="space-y-2 rounded-lg border border-border bg-muted/30 p-3">
                        <div className="flex justify-between text-xs">
                          <span className="text-muted-foreground">
                            STT Model
                          </span>
                          <span className="font-mono">{voice.stt_model}</span>
                        </div>
                        <div className="flex justify-between text-xs">
                          <span className="text-muted-foreground">
                            TTS Model
                          </span>
                          <span className="font-mono">{voice.tts_model}</span>
                        </div>
                        <div className="flex justify-between text-xs">
                          <span className="text-muted-foreground">
                            TTS Voice
                          </span>
                          <span className="font-mono">{voice.tts_voice}</span>
                        </div>
                      </div>

                      <Separator />

                      {/* Wake word toggle */}
                      <div className="flex items-center justify-between">
                        <div className="space-y-0.5">
                          <p className="text-sm font-medium">Wake Word</p>
                          <p className="text-xs text-muted-foreground">
                            Activate voice with a wake word
                          </p>
                        </div>
                        <Switch
                          checked={voice.wake_word_enabled}
                          disabled={voiceLoading}
                          onCheckedChange={() =>
                            handleVoiceAction("toggle_wake_word")
                          }
                        />
                      </div>

                      {/* Hotkey toggle */}
                      <div className="flex items-center justify-between">
                        <div className="space-y-0.5">
                          <p className="text-sm font-medium">Hotkey</p>
                          <p className="text-xs text-muted-foreground">
                            Use a keyboard shortcut to activate voice
                          </p>
                        </div>
                        <Switch
                          checked={voice.hotkey_enabled}
                          disabled={voiceLoading}
                          onCheckedChange={() =>
                            handleVoiceAction("toggle_hotkey")
                          }
                        />
                      </div>

                      {/* Push-to-talk info */}
                      <div className="rounded-lg border border-border bg-muted/30 p-3 text-xs text-muted-foreground">
                        <strong className="text-foreground">
                          Push-to-Talk:
                        </strong>{" "}
                        Hold the configured hotkey to speak. Release to send.
                      </div>

                      <Separator />

                      {/* Start / Stop buttons */}
                      <div className="flex gap-3">
                        <Button
                          variant="outline"
                          onClick={() => handleVoiceAction("start")}
                          disabled={
                            voiceLoading || voice.state === "running"
                          }
                          className="flex-1"
                        >
                          {voiceLoading ? (
                            <Loader2 className="mr-2 size-4 animate-spin" />
                          ) : (
                            <Play className="mr-2 size-4" />
                          )}
                          Start Pipeline
                        </Button>
                        <Button
                          variant="outline"
                          onClick={() => handleVoiceAction("stop")}
                          disabled={
                            voiceLoading || voice.state !== "running"
                          }
                          className="flex-1"
                        >
                          <Square className="mr-2 size-4" />
                          Stop Pipeline
                        </Button>
                      </div>
                    </>
                  ) : (
                    <p className="text-sm text-muted-foreground">
                      Voice status unavailable. Is the backend running?
                    </p>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            {/* ============================================================ */}
            {/* DESKTOP TAB                                                   */}
            {/* ============================================================ */}
            <TabsContent value="desktop" className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-base">
                    <Monitor className="size-4 text-violet-400" />
                    Desktop Automation
                  </CardTitle>
                  <CardDescription>
                    Control how ARA interacts with your desktop environment
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  {/* Enabled toggle */}
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <p className="text-sm font-medium">Enabled</p>
                      <p className="text-xs text-muted-foreground">
                        Allow ARA to perform desktop automation actions
                      </p>
                    </div>
                    <Switch
                      checked={desktopEnabled}
                      onCheckedChange={(v) => {
                        setDesktopEnabled(v);
                        saveDesktop(v, desktopConfirmation);
                      }}
                    />
                  </div>

                  {/* Require confirmation */}
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <p className="text-sm font-medium">
                        Require Confirmation
                      </p>
                      <p className="text-xs text-muted-foreground">
                        Ask for approval before executing desktop commands
                      </p>
                    </div>
                    <Switch
                      checked={desktopConfirmation}
                      onCheckedChange={(v) => {
                        setDesktopConfirmation(v);
                        saveDesktop(desktopEnabled, v);
                      }}
                    />
                  </div>

                  <Separator />

                  {/* Timeout display */}
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Clock className="size-4 text-muted-foreground" />
                      <span className="text-sm text-muted-foreground">
                        Command Timeout
                      </span>
                    </div>
                    <span className="font-mono text-sm">
                      {settings?.desktop.command_timeout_seconds ?? "—"}s
                    </span>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            {/* ============================================================ */}
            {/* INTERNET TAB                                                  */}
            {/* ============================================================ */}
            <TabsContent value="internet" className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-base">
                    <Globe className="size-4 text-violet-400" />
                    Internet Access
                  </CardTitle>
                  <CardDescription>
                    Configure web search and internet connectivity
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  {/* Internet enabled toggle */}
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      {internetEnabled ? (
                        <Wifi className="size-5 text-emerald-400" />
                      ) : (
                        <WifiOff className="size-5 text-zinc-500" />
                      )}
                      <div className="space-y-0.5">
                        <p className="text-sm font-medium">Internet Enabled</p>
                        <p className="text-xs text-muted-foreground">
                          Allow ARA to search the web and fetch content
                        </p>
                      </div>
                    </div>
                    <Switch
                      checked={internetEnabled}
                      onCheckedChange={(v) => {
                        setInternetEnabled(v);
                        saveInternet(v);
                      }}
                    />
                  </div>

                  <Separator />

                  {/* Cache TTL */}
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Clock className="size-4 text-muted-foreground" />
                      <span className="text-sm text-muted-foreground">
                        Cache TTL
                      </span>
                    </div>
                    <span className="font-mono text-sm">
                      {settings?.internet.web_cache_ttl_s ?? "—"}s
                    </span>
                  </div>

                  {/* SearXNG URL */}
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Globe className="size-4 text-muted-foreground" />
                      <span className="text-sm text-muted-foreground">
                        SearXNG URL
                      </span>
                    </div>
                    <span className="max-w-[200px] truncate font-mono text-sm">
                      {settings?.internet.searxng_url ?? "—"}
                    </span>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            {/* ============================================================ */}
            {/* ABOUT TAB                                                     */}
            {/* ============================================================ */}
            <TabsContent value="about" className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-base">
                    <Info className="size-4 text-violet-400" />
                    About ARA
                  </CardTitle>
                  <CardDescription>
                    System status and version information
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  {health ? (
                    <>
                      {/* Version */}
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-muted-foreground">
                          Version
                        </span>
                        <Badge variant="outline">{health.version}</Badge>
                      </div>

                      {/* Health status */}
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-muted-foreground">
                          Status
                        </span>
                        <span className="flex items-center gap-1.5">
                          <span
                            className={`size-2 rounded-full ${
                              health.status === "ok"
                                ? "bg-emerald-500"
                                : "bg-red-500"
                            }`}
                          />
                          <span
                            className={`text-sm font-medium ${
                              health.status === "ok"
                                ? "text-emerald-400"
                                : "text-red-400"
                            }`}
                          >
                            {health.status === "ok" ? "Healthy" : health.status}
                          </span>
                        </span>
                      </div>

                      {/* Uptime */}
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-muted-foreground">
                          Uptime
                        </span>
                        <span className="font-mono text-sm">
                          {formatUptime(health.uptime_s)}
                        </span>
                      </div>

                      <Separator />

                      {/* Models loaded */}
                      <div className="space-y-2">
                        <span className="text-sm font-medium">
                          Models Loaded
                        </span>
                        {health.models_loaded.length > 0 ? (
                          <div className="flex flex-wrap gap-2">
                            {health.models_loaded.map((model) => (
                              <Badge
                                key={model}
                                variant="secondary"
                                className="font-mono text-xs"
                              >
                                {model}
                              </Badge>
                            ))}
                          </div>
                        ) : (
                          <p className="text-xs text-muted-foreground">
                            No models loaded
                          </p>
                        )}
                      </div>
                    </>
                  ) : (
                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                      <Shield className="size-4" />
                      Health data unavailable. Is the backend running?
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        )}
      </div>
    </div>
  );
}
