"use client";

import { useCallback, useEffect, useState } from "react";
import {
  Brain,
  Search,
  FileText,
  Calendar,
  MessageSquare,
  BookOpen,
  Rocket,
  Dumbbell,
} from "lucide-react";

import { searchMemory, fetchMemoryStats } from "@/lib/api";
import type {
  MemoryChunkItem,
  MemorySearchResponse,
  MemoryStatsResponse,
} from "@/types/api";

import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";

// ---------------------------------------------------------------------------
// Source type helpers
// ---------------------------------------------------------------------------

const SOURCE_TYPES = ["All", "Diary", "Conversation", "Study", "Startup"] as const;

const SOURCE_COLORS: Record<string, string> = {
  diary: "bg-amber-500/20 text-amber-400 border-amber-500/30",
  conversation: "bg-blue-500/20 text-blue-400 border-blue-500/30",
  study: "bg-emerald-500/20 text-emerald-400 border-emerald-500/30",
  startup: "bg-purple-500/20 text-purple-400 border-purple-500/30",
};

const SOURCE_ICONS: Record<string, React.ReactNode> = {
  diary: <BookOpen className="size-3" />,
  conversation: <MessageSquare className="size-3" />,
  study: <FileText className="size-3" />,
  startup: <Rocket className="size-3" />,
};

function getSourceColor(source: string) {
  const key = source.toLowerCase();
  for (const [k, v] of Object.entries(SOURCE_COLORS)) {
    if (key.includes(k)) return v;
  }
  return "bg-zinc-500/20 text-zinc-400 border-zinc-500/30";
}

function getSourceIcon(source: string) {
  const key = source.toLowerCase();
  for (const [k, v] of Object.entries(SOURCE_ICONS)) {
    if (key.includes(k)) return v;
  }
  return <FileText className="size-3" />;
}

function getSourceLabel(source: string) {
  const key = source.toLowerCase();
  for (const label of ["Diary", "Conversation", "Study", "Startup"]) {
    if (key.includes(label.toLowerCase())) return label;
  }
  return source;
}

// ---------------------------------------------------------------------------
// Skeleton card for loading state
// ---------------------------------------------------------------------------

function SkeletonCard() {
  return (
    <Card className="border-border/50 bg-card/60 animate-pulse">
      <CardContent className="space-y-3 py-4">
        <div className="flex items-center gap-2">
          <div className="h-5 w-20 rounded-full bg-muted" />
          <div className="h-4 w-32 rounded bg-muted" />
        </div>
        <div className="space-y-2">
          <div className="h-3 w-full rounded bg-muted" />
          <div className="h-3 w-4/5 rounded bg-muted" />
          <div className="h-3 w-3/5 rounded bg-muted" />
        </div>
        <div className="flex items-center justify-between">
          <div className="h-3 w-24 rounded bg-muted" />
          <div className="h-2 w-16 rounded-full bg-muted" />
        </div>
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Memory chunk card
// ---------------------------------------------------------------------------

function ChunkCard({ chunk }: { chunk: MemoryChunkItem }) {
  const [expanded, setExpanded] = useState(false);

  const preview =
    chunk.text.length > 200 && !expanded
      ? chunk.text.slice(0, 200) + "..."
      : chunk.text;

  const scorePercent = Math.round(chunk.score * 100);
  const scoreColor =
    scorePercent >= 80
      ? "bg-emerald-500"
      : scorePercent >= 60
        ? "bg-amber-500"
        : scorePercent >= 40
          ? "bg-orange-500"
          : "bg-red-500";

  return (
    <Card className="group border-border/50 bg-card/80 transition-colors hover:border-border">
      <CardContent className="space-y-3 py-4">
        {/* Top row: source badge + section title */}
        <div className="flex flex-wrap items-center gap-2">
          <Badge
            variant="outline"
            className={`gap-1 border ${getSourceColor(chunk.source)}`}
          >
            {getSourceIcon(chunk.source)}
            {getSourceLabel(chunk.source)}
          </Badge>
          {chunk.section_title && (
            <span className="text-sm font-medium text-foreground/80">
              {chunk.section_title}
            </span>
          )}
        </div>

        {/* Text preview */}
        <button
          type="button"
          onClick={() => chunk.text.length > 200 && setExpanded(!expanded)}
          className="block w-full text-left"
        >
          <p className="text-sm leading-relaxed text-muted-foreground whitespace-pre-wrap">
            {preview}
          </p>
          {chunk.text.length > 200 && (
            <span className="mt-1 text-xs text-muted-foreground/60">
              {expanded ? "Show less" : "Click to expand"}
            </span>
          )}
        </button>

        {/* Score bar */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground/70">Relevance</span>
          <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
            <div
              className={`h-full rounded-full transition-all ${scoreColor}`}
              style={{ width: `${scorePercent}%` }}
            />
          </div>
          <span className="text-xs font-mono text-muted-foreground/70">
            {scorePercent}%
          </span>
        </div>

        {/* Footer: source path + date */}
        <div className="flex items-center justify-between text-xs text-muted-foreground/50">
          <span className="truncate max-w-[70%]" title={chunk.source_file}>
            {chunk.source_file}
          </span>
          {chunk.date && (
            <span className="flex items-center gap-1">
              <Calendar className="size-3" />
              {chunk.date}
            </span>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Stats panel
// ---------------------------------------------------------------------------

function StatsPanel({ stats }: { stats: MemoryStatsResponse | null }) {
  if (!stats) return null;

  return (
    <Card className="border-border/50 bg-card/60">
      <CardContent className="py-4">
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-2">
            <Brain className="size-4 text-indigo-400" />
            <span className="text-sm font-medium">
              {stats.total_chunks.toLocaleString()}
            </span>
            <span className="text-xs text-muted-foreground">chunks</span>
          </div>

          <Separator orientation="vertical" className="h-4" />

          {Object.entries(stats.sources).map(([source, count]) => (
            <div key={source} className="flex items-center gap-1.5">
              <Badge
                variant="outline"
                className={`gap-1 border text-xs ${getSourceColor(source)}`}
              >
                {getSourceIcon(source)}
                {getSourceLabel(source)}
              </Badge>
              <span className="text-xs font-mono text-muted-foreground">
                {count.toLocaleString()}
              </span>
            </div>
          ))}

          {stats.graph_entities > 0 && (
            <>
              <Separator orientation="vertical" className="h-4" />
              <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                <span className="font-medium text-foreground/70">
                  {stats.graph_entities}
                </span>{" "}
                entities
                <span className="mx-0.5">/</span>
                <span className="font-medium text-foreground/70">
                  {stats.graph_relations}
                </span>{" "}
                relations
              </div>
            </>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export default function MemoryPage() {
  const [query, setQuery] = useState("");
  const [sourceFilter, setSourceFilter] = useState("All");
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");

  const [results, setResults] = useState<MemoryChunkItem[]>([]);
  const [totalCandidates, setTotalCandidates] = useState(0);
  const [retrievalTime, setRetrievalTime] = useState(0);
  const [stats, setStats] = useState<MemoryStatsResponse | null>(null);

  const [loading, setLoading] = useState(false);
  const [searched, setSearched] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch stats on mount
  useEffect(() => {
    fetchMemoryStats()
      .then(setStats)
      .catch(() => {
        // stats are optional, don't block the page
      });
  }, []);

  const handleSearch = useCallback(async () => {
    const trimmed = query.trim();
    if (!trimmed) return;

    setLoading(true);
    setError(null);
    setSearched(true);

    try {
      const res: MemorySearchResponse = await searchMemory({
        query: trimmed,
        top_k: 20,
        source_filter: sourceFilter === "All" ? undefined : sourceFilter.toLowerCase(),
        date_from: dateFrom || undefined,
        date_to: dateTo || undefined,
      });
      setResults(res.results);
      setTotalCandidates(res.total_candidates);
      setRetrievalTime(res.retrieval_time_ms);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Search failed");
      setResults([]);
    } finally {
      setLoading(false);
    }
  }, [query, sourceFilter, dateFrom, dateTo]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") handleSearch();
  };

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="border-b border-border/50 px-6 py-4">
        <div className="flex items-center gap-3">
          <div className="flex size-9 items-center justify-center rounded-lg bg-indigo-500/15">
            <Brain className="size-5 text-indigo-400" />
          </div>
          <div className="flex items-center gap-3">
            <h1 className="text-xl font-semibold tracking-tight">Memory Browser</h1>
            {stats && (
              <Badge variant="secondary" className="font-mono text-xs">
                {stats.total_chunks.toLocaleString()} chunks
              </Badge>
            )}
          </div>
        </div>
      </div>

      <ScrollArea className="flex-1">
        <div className="mx-auto w-full max-w-4xl space-y-4 p-6">
          {/* Stats */}
          <StatsPanel stats={stats} />

          {/* Search bar */}
          <Card className="border-border/50 bg-card/60">
            <CardContent className="space-y-3 py-4">
              {/* Query + button */}
              <div className="flex gap-2">
                <div className="relative flex-1">
                  <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
                  <Input
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="Search your memories..."
                    className="pl-9"
                  />
                </div>
                <Button onClick={handleSearch} disabled={loading || !query.trim()}>
                  <Search className="size-4" />
                  Search
                </Button>
              </div>

              {/* Filters row */}
              <div className="flex flex-wrap items-center gap-3">
                {/* Source filter */}
                <div className="flex items-center gap-2">
                  <label className="text-xs text-muted-foreground">Source</label>
                  <select
                    value={sourceFilter}
                    onChange={(e) => setSourceFilter(e.target.value)}
                    className="h-8 rounded-md border border-input bg-transparent px-2 text-sm text-foreground outline-none focus:border-ring focus:ring-[3px] focus:ring-ring/50 dark:bg-input/30"
                  >
                    {SOURCE_TYPES.map((s) => (
                      <option key={s} value={s}>
                        {s}
                      </option>
                    ))}
                  </select>
                </div>

                <Separator orientation="vertical" className="h-5" />

                {/* Date range */}
                <div className="flex items-center gap-2">
                  <Calendar className="size-3.5 text-muted-foreground" />
                  <Input
                    value={dateFrom}
                    onChange={(e) => setDateFrom(e.target.value)}
                    placeholder="YYYY-MM-DD"
                    className="h-8 w-32 text-xs"
                  />
                  <span className="text-xs text-muted-foreground">to</span>
                  <Input
                    value={dateTo}
                    onChange={(e) => setDateTo(e.target.value)}
                    placeholder="YYYY-MM-DD"
                    className="h-8 w-32 text-xs"
                  />
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Error */}
          {error && (
            <Card className="border-red-500/30 bg-red-500/10">
              <CardContent className="py-3">
                <p className="text-sm text-red-400">{error}</p>
              </CardContent>
            </Card>
          )}

          {/* Results meta */}
          {searched && !loading && results.length > 0 && (
            <div className="flex items-center justify-between text-xs text-muted-foreground">
              <span>
                {results.length} results from {totalCandidates.toLocaleString()}{" "}
                candidates
              </span>
              <span>{Math.round(retrievalTime)}ms</span>
            </div>
          )}

          {/* Loading state */}
          {loading && (
            <div className="grid gap-3 sm:grid-cols-1">
              {Array.from({ length: 4 }).map((_, i) => (
                <SkeletonCard key={i} />
              ))}
            </div>
          )}

          {/* Results */}
          {!loading && results.length > 0 && (
            <div className="grid gap-3 sm:grid-cols-1">
              {results.map((chunk) => (
                <ChunkCard key={chunk.chunk_id} chunk={chunk} />
              ))}
            </div>
          )}

          {/* Empty state: searched but no results */}
          {!loading && searched && results.length === 0 && !error && (
            <div className="flex flex-col items-center justify-center gap-3 py-16 text-center">
              <div className="flex size-14 items-center justify-center rounded-2xl bg-muted">
                <Search className="size-6 text-muted-foreground" />
              </div>
              <p className="text-sm font-medium text-muted-foreground">
                No memories found
              </p>
              <p className="text-xs text-muted-foreground/60">
                Try a different query or adjust filters
              </p>
            </div>
          )}

          {/* Initial state: no search yet */}
          {!loading && !searched && (
            <div className="flex flex-col items-center justify-center gap-3 py-20 text-center">
              <div className="flex size-16 items-center justify-center rounded-2xl bg-indigo-500/10">
                <Brain className="size-8 text-indigo-400/60" />
              </div>
              <p className="text-base font-medium text-muted-foreground">
                Search your memories...
              </p>
              <p className="max-w-sm text-xs text-muted-foreground/60">
                Query across diary entries, conversations, study notes, and more.
                Results are ranked by semantic relevance.
              </p>
            </div>
          )}
        </div>
      </ScrollArea>
    </div>
  );
}
