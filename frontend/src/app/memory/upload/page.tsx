"use client";

import { useCallback, useState } from "react";
import { useDropzone, type FileRejection } from "react-dropzone";
import {
  Upload,
  FileText,
  X,
  CheckCircle,
  AlertCircle,
  Loader2,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface SelectedFile {
  id: string;
  file: File;
  progress: number;
  status: "pending" | "uploading" | "done" | "error";
  error?: string;
}

interface RejectedFile {
  name: string;
  reason: string;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MAX_SIZE = 20 * 1024 * 1024; // 20 MB per file

const ACCEPT = {
  "text/markdown": [".md"],
  "text/plain": [".txt"],
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function typeBadgeLabel(name: string): string {
  if (name.endsWith(".md")) return "Markdown";
  if (name.endsWith(".txt")) return "Text";
  return "File";
}

let fileIdCounter = 0;
function nextId(): string {
  fileIdCounter += 1;
  return `file-${fileIdCounter}-${Date.now()}`;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function DiaryUploadPage() {
  const [files, setFiles] = useState<SelectedFile[]>([]);
  const [rejections, setRejections] = useState<RejectedFile[]>([]);
  const [ingesting, setIngesting] = useState(false);
  const [completed, setCompleted] = useState(false);
  const [currentIndex, setCurrentIndex] = useState(0);

  // ---- Dropzone ------------------------------------------------------------

  const onDrop = useCallback(
    (accepted: File[], rejected: FileRejection[]) => {
      const newFiles: SelectedFile[] = accepted.map((file) => ({
        id: nextId(),
        file,
        progress: 0,
        status: "pending" as const,
      }));
      setFiles((prev) => [...prev, ...newFiles]);

      const newRejections: RejectedFile[] = rejected.map((r) => ({
        name: r.file.name,
        reason: r.errors.map((e) => e.message).join(", "),
      }));
      if (newRejections.length > 0) {
        setRejections((prev) => [...prev, ...newRejections]);
      }
    },
    [],
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ACCEPT,
    maxSize: MAX_SIZE,
    multiple: true,
    disabled: ingesting,
  });

  // ---- Remove file ---------------------------------------------------------

  const removeFile = useCallback(
    (id: string) => {
      if (ingesting) return;
      setFiles((prev) => prev.filter((f) => f.id !== id));
    },
    [ingesting],
  );

  const dismissRejection = useCallback((index: number) => {
    setRejections((prev) => prev.filter((_, i) => i !== index));
  }, []);

  // ---- Simulated ingestion -------------------------------------------------

  const startIngestion = useCallback(() => {
    if (files.length === 0 || ingesting) return;

    setIngesting(true);
    setCompleted(false);
    setCurrentIndex(0);

    // Reset all files to pending / 0
    setFiles((prev) =>
      prev.map((f) => ({ ...f, progress: 0, status: "pending" as const })),
    );

    // Simulate sequential processing
    const totalFiles = files.length;

    function processFile(index: number) {
      if (index >= totalFiles) {
        setIngesting(false);
        setCompleted(true);
        return;
      }

      setCurrentIndex(index);
      setFiles((prev) =>
        prev.map((f, i) =>
          i === index ? { ...f, status: "uploading" as const } : f,
        ),
      );

      // Simulate progress increments
      let progress = 0;
      const interval = setInterval(() => {
        progress += Math.random() * 20 + 10;
        if (progress >= 100) {
          progress = 100;
          clearInterval(interval);
          setFiles((prev) =>
            prev.map((f, i) =>
              i === index
                ? { ...f, progress: 100, status: "done" as const }
                : f,
            ),
          );
          // Next file after a short pause
          setTimeout(() => processFile(index + 1), 300);
        } else {
          setFiles((prev) =>
            prev.map((f, i) =>
              i === index ? { ...f, progress: Math.min(progress, 99) } : f,
            ),
          );
        }
      }, 200);
    }

    processFile(0);
  }, [files, ingesting]);

  // ---- Derived state -------------------------------------------------------

  const totalSize = files.reduce((sum, f) => sum + f.file.size, 0);
  const overallProgress =
    files.length > 0
      ? Math.round(files.reduce((sum, f) => sum + f.progress, 0) / files.length)
      : 0;
  const doneCount = files.filter((f) => f.status === "done").length;
  const hasPendingFiles = files.some((f) => f.status === "pending");

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="border-b border-border/50 px-6 py-4">
        <div className="flex items-center gap-3">
          <div className="flex size-9 items-center justify-center rounded-lg bg-indigo-500/15">
            <Upload className="size-5 text-indigo-400" />
          </div>
          <h1 className="text-xl font-semibold tracking-tight">
            Upload Diaries
          </h1>
        </div>
      </div>

      <div className="flex-1 overflow-auto">
        <div className="mx-auto w-full max-w-3xl space-y-4 p-6">
          {/* Drop zone */}
          <div
            {...getRootProps()}
            className={cn(
              "flex cursor-pointer flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed px-6 py-14 transition-colors",
              ingesting && "pointer-events-none opacity-50",
              isDragActive
                ? "border-indigo-500 bg-indigo-500/10"
                : "border-border/50 hover:border-border hover:bg-card/40",
            )}
          >
            <input {...getInputProps()} />
            <div
              className={cn(
                "flex size-14 items-center justify-center rounded-2xl transition-colors",
                isDragActive ? "bg-indigo-500/20" : "bg-muted",
              )}
            >
              <Upload
                className={cn(
                  "size-6 transition-colors",
                  isDragActive ? "text-indigo-400" : "text-muted-foreground",
                )}
              />
            </div>
            <div className="text-center">
              <p className="text-sm font-medium text-foreground/80">
                {isDragActive
                  ? "Drop files here..."
                  : "Drop files here or click to browse"}
              </p>
              <p className="mt-1 text-xs text-muted-foreground/60">
                Accepts .md and .txt files (max 20 MB each)
              </p>
            </div>
          </div>

          {/* Rejection errors */}
          {rejections.length > 0 && (
            <div className="space-y-2">
              {rejections.map((r, i) => (
                <Card key={i} className="border-red-500/30 bg-red-500/10">
                  <CardContent className="flex items-center gap-3 py-3">
                    <AlertCircle className="size-4 shrink-0 text-red-400" />
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-sm font-medium text-red-300">
                        {r.name}
                      </p>
                      <p className="text-xs text-red-400/70">{r.reason}</p>
                    </div>
                    <Button
                      variant="ghost"
                      size="icon-xs"
                      className="shrink-0 text-red-400 hover:text-red-300"
                      onClick={() => dismissRejection(i)}
                    >
                      <X className="size-3" />
                    </Button>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}

          {/* File list */}
          {files.length > 0 && (
            <Card className="border-border/50 bg-card/60">
              <CardContent className="space-y-3 py-4">
                {/* Summary */}
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-foreground/80">
                    {files.length} file{files.length !== 1 ? "s" : ""} selected
                  </span>
                  <span className="text-xs text-muted-foreground">
                    {formatSize(totalSize)} total
                  </span>
                </div>

                <Separator />

                {/* Individual files */}
                <div className="space-y-2">
                  {files.map((f) => (
                    <div
                      key={f.id}
                      className="flex items-center gap-3 rounded-lg border border-border/30 bg-background/40 px-3 py-2"
                    >
                      {/* Icon */}
                      <div className="flex size-8 shrink-0 items-center justify-center rounded-md bg-muted">
                        {f.status === "done" ? (
                          <CheckCircle className="size-4 text-emerald-400" />
                        ) : f.status === "uploading" ? (
                          <Loader2 className="size-4 animate-spin text-indigo-400" />
                        ) : (
                          <FileText className="size-4 text-muted-foreground" />
                        )}
                      </div>

                      {/* File info */}
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-2">
                          <p className="truncate text-sm font-medium text-foreground/80">
                            {f.file.name}
                          </p>
                          <Badge
                            variant="secondary"
                            className="shrink-0 text-[10px] px-1.5 py-0 h-4"
                          >
                            {typeBadgeLabel(f.file.name)}
                          </Badge>
                        </div>
                        <p className="text-[10px] text-muted-foreground">
                          {formatSize(f.file.size)}
                        </p>

                        {/* Per-file progress bar (during ingestion) */}
                        {(f.status === "uploading" || f.status === "done") && (
                          <Progress
                            value={f.progress}
                            className={cn(
                              "mt-1 h-1",
                              f.status === "done" &&
                                "[&>[data-slot=progress-indicator]]:bg-emerald-500",
                            )}
                          />
                        )}
                      </div>

                      {/* Remove button (only when not ingesting) */}
                      {!ingesting && f.status !== "done" && (
                        <Button
                          variant="ghost"
                          size="icon-xs"
                          className="shrink-0 text-muted-foreground hover:text-foreground"
                          onClick={() => removeFile(f.id)}
                          aria-label={`Remove ${f.file.name}`}
                        >
                          <X className="size-3" />
                        </Button>
                      )}
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Ingestion progress section */}
          {ingesting && (
            <Card className="border-indigo-500/20 bg-indigo-500/5">
              <CardContent className="space-y-3 py-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-foreground/80">
                    Processing file {currentIndex + 1} of {files.length}...
                  </span>
                  <span className="text-xs font-mono text-muted-foreground">
                    {overallProgress}%
                  </span>
                </div>
                <Progress
                  value={overallProgress}
                  className="[&>[data-slot=progress-indicator]]:bg-indigo-500"
                />
              </CardContent>
            </Card>
          )}

          {/* Completion state */}
          {completed && !ingesting && (
            <Card className="border-emerald-500/20 bg-emerald-500/5">
              <CardContent className="flex items-center gap-3 py-4">
                <div className="flex size-10 items-center justify-center rounded-full bg-emerald-500/20">
                  <CheckCircle className="size-5 text-emerald-400" />
                </div>
                <div>
                  <p className="text-sm font-medium text-emerald-300">
                    {doneCount} file{doneCount !== 1 ? "s" : ""} ingested
                    successfully
                  </p>
                  <p className="text-xs text-emerald-400/60">
                    Your diary entries are now searchable in memory.
                  </p>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Start Ingestion button */}
          {files.length > 0 && !completed && (
            <div className="flex justify-end">
              <Button
                onClick={startIngestion}
                disabled={ingesting || !hasPendingFiles}
                className={cn(
                  "gap-2",
                  !ingesting &&
                    "bg-indigo-600 text-white hover:bg-indigo-500 shadow-md shadow-indigo-500/20",
                )}
              >
                {ingesting ? (
                  <>
                    <Loader2 className="size-4 animate-spin" />
                    Ingesting...
                  </>
                ) : (
                  <>
                    <Upload className="size-4" />
                    Start Ingestion
                  </>
                )}
              </Button>
            </div>
          )}

          {/* Reset after completion */}
          {completed && !ingesting && (
            <div className="flex justify-end">
              <Button
                variant="outline"
                onClick={() => {
                  setFiles([]);
                  setRejections([]);
                  setCompleted(false);
                  setCurrentIndex(0);
                }}
              >
                Upload More
              </Button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
