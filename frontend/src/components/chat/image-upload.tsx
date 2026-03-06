"use client";

import { useCallback, useEffect, useMemo } from "react";
import { useDropzone } from "react-dropzone";
import { ImagePlus, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface ImageUploadProps {
  onImageSelected: (file: File) => void;
  onImageRemoved: () => void;
  selectedImage: File | null;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const MAX_SIZE = 10 * 1024 * 1024; // 10 MB

const ACCEPT = {
  "image/jpeg": [".jpg", ".jpeg"],
  "image/png": [".png"],
  "image/webp": [".webp"],
  "image/gif": [".gif"],
};

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function ImageUpload({
  onImageSelected,
  onImageRemoved,
  selectedImage,
}: ImageUploadProps) {
  // ---- Dropzone config -----------------------------------------------------

  const onDrop = useCallback(
    (accepted: File[]) => {
      if (accepted.length > 0) {
        onImageSelected(accepted[0]);
      }
    },
    [onImageSelected],
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ACCEPT,
    maxSize: MAX_SIZE,
    multiple: false,
    noClick: !!selectedImage, // disable click when already have image
  });

  // ---- Clipboard paste support ---------------------------------------------

  useEffect(() => {
    function handlePaste(e: ClipboardEvent) {
      const items = e.clipboardData?.items;
      if (!items) return;

      for (const item of Array.from(items)) {
        if (item.type.startsWith("image/")) {
          const file = item.getAsFile();
          if (file) {
            e.preventDefault();
            onImageSelected(file);
            return;
          }
        }
      }
    }

    document.addEventListener("paste", handlePaste);
    return () => document.removeEventListener("paste", handlePaste);
  }, [onImageSelected]);

  // ---- Preview URL ---------------------------------------------------------

  const previewUrl = useMemo(() => {
    if (!selectedImage) return null;
    return URL.createObjectURL(selectedImage);
  }, [selectedImage]);

  // Clean up object URL on unmount or image change
  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  // ---- If an image is selected, show compact inline preview ----------------

  if (selectedImage && previewUrl) {
    return (
      <div className="flex items-center gap-2 rounded-lg border border-border/50 bg-card/60 px-2 py-1.5">
        <div className="relative size-16 shrink-0 overflow-hidden rounded-md border border-border/30">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={previewUrl}
            alt="Preview"
            className="size-full object-cover"
          />
          <Button
            type="button"
            variant="ghost"
            size="icon-xs"
            className="absolute -right-0.5 -top-0.5 size-5 rounded-full bg-background/80 backdrop-blur-sm hover:bg-destructive hover:text-white"
            onClick={(e) => {
              e.stopPropagation();
              onImageRemoved();
            }}
            aria-label="Remove image"
          >
            <X className="size-3" />
          </Button>
        </div>
        <div className="min-w-0 flex-1">
          <p className="truncate text-xs font-medium text-foreground/80">
            {selectedImage.name}
          </p>
          <p className="text-[10px] text-muted-foreground">
            {formatSize(selectedImage.size)}
          </p>
        </div>
      </div>
    );
  }

  // ---- Drop zone (no image selected) ---------------------------------------

  return (
    <div
      {...getRootProps()}
      className={cn(
        "flex cursor-pointer items-center justify-center gap-2 rounded-lg border-2 border-dashed px-3 py-2 transition-colors",
        isDragActive
          ? "border-indigo-500 bg-indigo-500/10"
          : "border-border/50 hover:border-border hover:bg-card/40",
      )}
    >
      <input {...getInputProps()} />
      <ImagePlus
        className={cn(
          "size-4 transition-colors",
          isDragActive ? "text-indigo-400" : "text-muted-foreground",
        )}
      />
      <span className="text-xs text-muted-foreground">
        {isDragActive ? "Drop image here" : "Paste or drop an image"}
      </span>
    </div>
  );
}
