"use client";

import { useEffect, useState, useCallback } from "react";
import {
  Cpu,
  HardDrive,
  Zap,
  MemoryStick,
  Activity,
  Loader2,
  ArrowRight,
  RefreshCw,
} from "lucide-react";
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
import { Separator } from "@/components/ui/separator";
import { fetchModels, fetchVRAM, swapModel } from "@/lib/api";
import type {
  ModelItem,
  VRAMUsageResponse,
  ModelSwapResponse,
} from "@/types/api";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const ROLE_LABELS: Record<string, string> = {
  primary_llm: "Primary LLM",
  vision_always_on: "Vision (Always-On)",
  stt: "Speech-to-Text",
  tts: "Text-to-Speech",
  embeddings: "Embeddings",
  reranker: "Reranker",
  vad: "Voice Activity Detection",
  wake_word: "Wake Word",
};

function formatRole(role: string): string {
  return ROLE_LABELS[role] ?? role.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function ModelsPage() {
  const [models, setModels] = useState<ModelItem[]>([]);
  const [vram, setVram] = useState<VRAMUsageResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Swap state
  const [targetModel, setTargetModel] = useState("");
  const [swapping, setSwapping] = useState(false);
  const [swapResult, setSwapResult] = useState<ModelSwapResponse | null>(null);
  const [swapError, setSwapError] = useState<string | null>(null);

  const loadData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const [m, v] = await Promise.all([fetchModels(), fetchVRAM()]);
      setModels(m);
      setVram(v);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load models");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const handleSwap = async () => {
    if (!targetModel.trim()) return;
    try {
      setSwapping(true);
      setSwapError(null);
      setSwapResult(null);
      const result = await swapModel({ target_model: targetModel.trim() });
      setSwapResult(result);
      setTargetModel("");
      // Refresh data after swap
      await loadData();
    } catch (err) {
      setSwapError(err instanceof Error ? err.message : "Swap failed");
    } finally {
      setSwapping(false);
    }
  };

  const usedPercent = vram ? (vram.used_gb / vram.total_gb) * 100 : 0;

  return (
    <div className="min-h-screen bg-background p-6 md:p-8">
      <div className="mx-auto max-w-5xl space-y-8">
        {/* ---- Header ---------------------------------------------------- */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex size-10 items-center justify-center rounded-lg bg-indigo-500/10 text-indigo-400">
              <Cpu className="size-5" />
            </div>
            <div>
              <h1 className="text-2xl font-bold tracking-tight">
                Model Management
              </h1>
              <p className="text-sm text-muted-foreground">
                Monitor loaded models, VRAM usage, and swap models on the fly
              </p>
            </div>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={loadData}
            disabled={loading}
          >
            <RefreshCw
              className={`mr-2 size-4 ${loading ? "animate-spin" : ""}`}
            />
            Refresh
          </Button>
        </div>

        {error && (
          <Card className="border-red-500/30 bg-red-500/5">
            <CardContent className="py-4 text-sm text-red-400">
              {error}
            </CardContent>
          </Card>
        )}

        {/* ---- VRAM Usage ------------------------------------------------ */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <Zap className="size-4 text-violet-400" />
              VRAM Usage
            </CardTitle>
            <CardDescription>GPU memory allocation overview</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {vram ? (
              <>
                {/* Bar */}
                <div className="relative h-5 w-full overflow-hidden rounded-full bg-muted">
                  <div
                    className="absolute inset-y-0 left-0 rounded-full bg-gradient-to-r from-indigo-500 to-violet-500 transition-all duration-500"
                    style={{ width: `${usedPercent}%` }}
                  />
                </div>

                {/* Labels */}
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">
                    <span className="font-medium text-foreground">
                      {vram.used_gb.toFixed(1)} GB
                    </span>{" "}
                    / {vram.total_gb.toFixed(1)} GB used
                  </span>
                  <span
                    className={`font-mono font-medium ${
                      usedPercent > 90
                        ? "text-red-400"
                        : usedPercent > 70
                          ? "text-amber-400"
                          : "text-emerald-400"
                    }`}
                  >
                    {usedPercent.toFixed(1)}%
                  </span>
                </div>

                {/* Free vs Used legend */}
                <div className="flex gap-4 text-xs text-muted-foreground">
                  <span className="flex items-center gap-1.5">
                    <span className="size-2 rounded-full bg-gradient-to-r from-indigo-500 to-violet-500" />
                    Used ({vram.used_gb.toFixed(1)} GB)
                  </span>
                  <span className="flex items-center gap-1.5">
                    <span className="size-2 rounded-full bg-muted-foreground/30" />
                    Free ({vram.free_gb.toFixed(1)} GB)
                  </span>
                </div>
              </>
            ) : loading ? (
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Loader2 className="size-4 animate-spin" />
                Loading VRAM data...
              </div>
            ) : null}
          </CardContent>
        </Card>

        {/* ---- Models List ----------------------------------------------- */}
        <div className="space-y-4">
          <h2 className="flex items-center gap-2 text-lg font-semibold">
            <Activity className="size-4 text-indigo-400" />
            Loaded Models
            {models.length > 0 && (
              <Badge variant="secondary" className="ml-1">
                {models.length}
              </Badge>
            )}
          </h2>

          {loading && models.length === 0 ? (
            <Card>
              <CardContent className="flex items-center justify-center py-12">
                <Loader2 className="mr-2 size-5 animate-spin text-muted-foreground" />
                <span className="text-muted-foreground">
                  Loading models...
                </span>
              </CardContent>
            </Card>
          ) : (
            <div className="grid gap-4 sm:grid-cols-2">
              {models.map((model) => (
                <Card
                  key={`${model.role}-${model.name}`}
                  className="transition-colors hover:border-indigo-500/30"
                >
                  <CardHeader className="pb-0">
                    <div className="flex items-start justify-between">
                      <div className="space-y-1">
                        <CardTitle className="text-sm font-semibold leading-tight">
                          {model.name}
                        </CardTitle>
                        <CardDescription className="text-xs">
                          {formatRole(model.role)}
                        </CardDescription>
                      </div>
                      {/* Loaded indicator */}
                      <span className="flex items-center gap-1.5">
                        <span
                          className={`size-2 rounded-full ${
                            model.loaded ? "bg-emerald-500" : "bg-zinc-500"
                          }`}
                        />
                        <span className="text-xs text-muted-foreground">
                          {model.loaded ? "Loaded" : "Unloaded"}
                        </span>
                      </span>
                    </div>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    {/* Location badge & memory */}
                    <div className="flex items-center gap-2">
                      {model.location === "gpu" ? (
                        <Badge className="bg-indigo-500/15 text-indigo-400 hover:bg-indigo-500/20">
                          <Zap className="mr-1 size-3" />
                          GPU
                        </Badge>
                      ) : (
                        <Badge
                          variant="secondary"
                          className="bg-zinc-500/15 text-zinc-400 hover:bg-zinc-500/20"
                        >
                          <HardDrive className="mr-1 size-3" />
                          CPU
                        </Badge>
                      )}

                      {model.location === "gpu" ? (
                        <span className="flex items-center gap-1 text-xs text-muted-foreground">
                          <MemoryStick className="size-3" />
                          {model.vram_gb.toFixed(1)} GB VRAM
                        </span>
                      ) : (
                        <span className="flex items-center gap-1 text-xs text-muted-foreground">
                          <MemoryStick className="size-3" />
                          {model.ram_mb} MB RAM
                        </span>
                      )}
                    </div>

                    {/* VRAM mini bar for GPU models */}
                    {model.location === "gpu" && vram && (
                      <div className="space-y-1">
                        <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
                          <div
                            className="h-full rounded-full bg-gradient-to-r from-indigo-500 to-violet-500 transition-all"
                            style={{
                              width: `${(model.vram_gb / vram.total_gb) * 100}%`,
                            }}
                          />
                        </div>
                        <p className="text-[10px] text-muted-foreground">
                          {((model.vram_gb / vram.total_gb) * 100).toFixed(1)}%
                          of total VRAM
                        </p>
                      </div>
                    )}

                    {/* Description */}
                    {model.description && (
                      <p className="text-xs leading-relaxed text-muted-foreground">
                        {model.description}
                      </p>
                    )}
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </div>

        <Separator />

        {/* ---- Model Swap ------------------------------------------------ */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <RefreshCw className="size-4 text-violet-400" />
              Model Swap
            </CardTitle>
            <CardDescription>
              Hot-swap the primary LLM to a different model
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex gap-3">
              <Input
                placeholder="Target model name (e.g. mistral-7b-instruct)"
                value={targetModel}
                onChange={(e) => setTargetModel(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleSwap();
                }}
                className="flex-1"
                disabled={swapping}
              />
              <Button
                onClick={handleSwap}
                disabled={swapping || !targetModel.trim()}
                className="bg-indigo-600 text-white hover:bg-indigo-700"
              >
                {swapping ? (
                  <>
                    <Loader2 className="mr-2 size-4 animate-spin" />
                    Swapping...
                  </>
                ) : (
                  <>
                    <RefreshCw className="mr-2 size-4" />
                    Swap Model
                  </>
                )}
              </Button>
            </div>

            {/* Swap result */}
            {swapResult && (
              <div className="rounded-lg border border-emerald-500/30 bg-emerald-500/5 p-4">
                <div className="flex items-center gap-2 text-sm">
                  <span className="font-medium text-muted-foreground">
                    {swapResult.previous_model}
                  </span>
                  <ArrowRight className="size-4 text-emerald-400" />
                  <span className="font-medium text-emerald-400">
                    {swapResult.current_model}
                  </span>
                </div>
                <p className="mt-1 text-xs text-muted-foreground">
                  {swapResult.message} &middot;{" "}
                  {swapResult.elapsed_ms.toFixed(0)}ms
                </p>
              </div>
            )}

            {/* Swap error */}
            {swapError && (
              <div className="rounded-lg border border-red-500/30 bg-red-500/5 p-3 text-sm text-red-400">
                {swapError}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
