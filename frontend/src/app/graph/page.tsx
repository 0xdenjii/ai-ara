"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  type Node,
  type Edge,
  type NodeProps,
  Handle,
  Position,
  useNodesState,
  useEdgesState,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { Network, Info, Loader2 } from "lucide-react";

import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  fetchGraphStats,
  searchGraphEntities,
  fetchGraphSubgraph,
  type GraphEntity,
  type GraphRelation,
} from "@/lib/api";

// ---------------------------------------------------------------------------
// Node type colors
// ---------------------------------------------------------------------------

const TYPE_STYLES: Record<string, { border: string; bg: string; badge: string }> = {
  Person: {
    border: "border-l-indigo-500",
    bg: "bg-indigo-500/10",
    badge: "bg-indigo-500/20 text-indigo-400 border-indigo-500/30",
  },
  Place: {
    border: "border-l-emerald-500",
    bg: "bg-emerald-500/10",
    badge: "bg-emerald-500/20 text-emerald-400 border-emerald-500/30",
  },
  Project: {
    border: "border-l-amber-500",
    bg: "bg-amber-500/10",
    badge: "bg-amber-500/20 text-amber-400 border-amber-500/30",
  },
  Topic: {
    border: "border-l-cyan-500",
    bg: "bg-cyan-500/10",
    badge: "bg-cyan-500/20 text-cyan-400 border-cyan-500/30",
  },
  Habit: {
    border: "border-l-pink-500",
    bg: "bg-pink-500/10",
    badge: "bg-pink-500/20 text-pink-400 border-pink-500/30",
  },
};

function getTypeStyle(type: string) {
  // Normalize: capitalize first letter for matching
  const normalized = type.charAt(0).toUpperCase() + type.slice(1).toLowerCase();
  return (
    TYPE_STYLES[normalized] ?? {
      border: "border-l-zinc-500",
      bg: "bg-zinc-500/10",
      badge: "bg-zinc-500/20 text-zinc-400 border-zinc-500/30",
    }
  );
}

// ---------------------------------------------------------------------------
// Custom graph node
// ---------------------------------------------------------------------------

type GraphNodeData = { label: string; type: string };

function GraphNode({ data }: NodeProps<Node<GraphNodeData>>) {
  const style = getTypeStyle(data.type);

  return (
    <>
      <Handle type="target" position={Position.Left} className="!bg-zinc-500 !border-zinc-600 !size-2" />
      <div
        className={`rounded-lg border-l-[3px] ${style.border} border border-border/60 bg-zinc-900/90 px-4 py-2.5 shadow-lg backdrop-blur-sm`}
      >
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-zinc-100">{data.label}</span>
          <Badge
            variant="outline"
            className={`border text-[10px] px-1.5 py-0 ${style.badge}`}
          >
            {data.type}
          </Badge>
        </div>
      </div>
      <Handle type="source" position={Position.Right} className="!bg-zinc-500 !border-zinc-600 !size-2" />
    </>
  );
}

// ---------------------------------------------------------------------------
// Demo / fallback data
// ---------------------------------------------------------------------------

const DEMO_NODES: Node<GraphNodeData>[] = [
  {
    id: "alex",
    type: "graphNode",
    position: { x: 50, y: 200 },
    data: { label: "Alex", type: "Person" },
  },
  {
    id: "sam",
    type: "graphNode",
    position: { x: 50, y: 400 },
    data: { label: "Sam", type: "Person" },
  },
  {
    id: "gym",
    type: "graphNode",
    position: { x: 350, y: 50 },
    data: { label: "Gym", type: "Habit" },
  },
  {
    id: "work",
    type: "graphNode",
    position: { x: 350, y: 200 },
    data: { label: "Work", type: "Project" },
  },
  {
    id: "javascript",
    type: "graphNode",
    position: { x: 350, y: 360 },
    data: { label: "JavaScript", type: "Topic" },
  },
  {
    id: "city",
    type: "graphNode",
    position: { x: 650, y: 280 },
    data: { label: "Hometown", type: "Place" },
  },
];

const DEMO_EDGES: Edge[] = [
  {
    id: "e-alex-work",
    source: "alex",
    target: "work",
    label: "WORKS_ON",
    animated: true,
    style: { stroke: "#6366f1" },
    labelStyle: { fill: "#a5b4fc", fontSize: 10, fontWeight: 500 },
    labelBgStyle: { fill: "#18181b", fillOpacity: 0.9 },
    labelBgPadding: [6, 3] as [number, number],
    labelBgBorderRadius: 4,
  },
  {
    id: "e-alex-city",
    source: "alex",
    target: "city",
    label: "LIVES_IN",
    style: { stroke: "#10b981" },
    labelStyle: { fill: "#6ee7b7", fontSize: 10, fontWeight: 500 },
    labelBgStyle: { fill: "#18181b", fillOpacity: 0.9 },
    labelBgPadding: [6, 3] as [number, number],
    labelBgBorderRadius: 4,
  },
  {
    id: "e-alex-sam",
    source: "alex",
    target: "sam",
    label: "KNOWS",
    style: { stroke: "#8b5cf6" },
    labelStyle: { fill: "#c4b5fd", fontSize: 10, fontWeight: 500 },
    labelBgStyle: { fill: "#18181b", fillOpacity: 0.9 },
    labelBgPadding: [6, 3] as [number, number],
    labelBgBorderRadius: 4,
  },
  {
    id: "e-alex-gym",
    source: "alex",
    target: "gym",
    label: "PRACTICES",
    style: { stroke: "#ec4899" },
    labelStyle: { fill: "#f9a8d4", fontSize: 10, fontWeight: 500 },
    labelBgStyle: { fill: "#18181b", fillOpacity: 0.9 },
    labelBgPadding: [6, 3] as [number, number],
    labelBgBorderRadius: 4,
  },
  {
    id: "e-alex-javascript",
    source: "alex",
    target: "javascript",
    label: "DISCUSSED",
    style: { stroke: "#06b6d4" },
    labelStyle: { fill: "#67e8f9", fontSize: 10, fontWeight: 500 },
    labelBgStyle: { fill: "#18181b", fillOpacity: 0.9 },
    labelBgPadding: [6, 3] as [number, number],
    labelBgBorderRadius: 4,
  },
  {
    id: "e-work-javascript",
    source: "work",
    target: "javascript",
    label: "USES",
    style: { stroke: "#f59e0b" },
    labelStyle: { fill: "#fcd34d", fontSize: 10, fontWeight: 500 },
    labelBgStyle: { fill: "#18181b", fillOpacity: 0.9 },
    labelBgPadding: [6, 3] as [number, number],
    labelBgBorderRadius: 4,
  },
];

// ---------------------------------------------------------------------------
// Helpers: transform API data to ReactFlow format
// ---------------------------------------------------------------------------

const EDGE_COLORS = [
  "#6366f1", "#10b981", "#8b5cf6", "#ec4899", "#06b6d4",
  "#f59e0b", "#ef4444", "#14b8a6", "#f97316", "#84cc16",
];

function entityToNode(entity: GraphEntity, index: number): Node<GraphNodeData> {
  // Lay out nodes in a grid-like pattern; fitView will adjust
  const cols = 4;
  const col = index % cols;
  const row = Math.floor(index / cols);
  const jitterX = (index * 37) % 60 - 30; // slight randomization for visual variety
  const jitterY = (index * 53) % 60 - 30;

  const typeLabel =
    entity.entity_type.charAt(0).toUpperCase() +
    entity.entity_type.slice(1).toLowerCase();

  return {
    id: entity.entity_id,
    type: "graphNode",
    position: { x: col * 280 + jitterX, y: row * 150 + jitterY },
    data: { label: entity.name, type: typeLabel },
  };
}

function relationToEdge(relation: GraphRelation, index: number): Edge {
  const color = EDGE_COLORS[index % EDGE_COLORS.length];
  return {
    id: `e-${relation.source_id}-${relation.target_id}-${index}`,
    source: relation.source_id,
    target: relation.target_id,
    label: relation.relation_type,
    animated: index === 0,
    style: { stroke: color },
    labelStyle: { fill: color, fontSize: 10, fontWeight: 500 },
    labelBgStyle: { fill: "#18181b", fillOpacity: 0.9 },
    labelBgPadding: [6, 3] as [number, number],
    labelBgBorderRadius: 4,
  };
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export default function GraphPage() {
  const [nodes, setNodes, onNodesChange] = useNodesState([] as Node<GraphNodeData>[]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([] as Edge[]);
  const [loading, setLoading] = useState(true);
  const [isDemo, setIsDemo] = useState(false);
  const [stats, setStats] = useState<{ entities: number; relations: number } | null>(null);

  const nodeTypes = useMemo(() => ({ graphNode: GraphNode }), []);

  // ---- Fetch real graph data on mount ------------------------------------

  useEffect(() => {
    let cancelled = false;

    async function loadGraph() {
      try {
        // 1. Get stats (also serves as a connectivity check)
        const graphStats = await fetchGraphStats();
        if (cancelled) return;
        setStats({
          entities: graphStats.entity_count,
          relations: graphStats.relation_count,
        });

        // If the graph is empty, fall back to demo
        if (graphStats.entity_count === 0) {
          setNodes(DEMO_NODES);
          setEdges(DEMO_EDGES);
          setIsDemo(true);
          setLoading(false);
          return;
        }

        // 2. Fetch entities (empty query = all, limited)
        const searchResult = await searchGraphEntities("", 50);
        if (cancelled) return;

        const entities = searchResult.entities;
        if (entities.length === 0) {
          setNodes(DEMO_NODES);
          setEdges(DEMO_EDGES);
          setIsDemo(true);
          setLoading(false);
          return;
        }

        // 3. Build node list from all entities
        const apiNodes = entities.map(entityToNode);

        // 4. Fetch subgraphs for relations. Query the first few entities to
        //    gather edges without hammering the backend.
        const entitiesToQuery = entities.slice(0, Math.min(entities.length, 10));
        const subgraphPromises = entitiesToQuery.map((e) =>
          fetchGraphSubgraph(e.entity_id).catch(() => null),
        );
        const subgraphs = await Promise.all(subgraphPromises);
        if (cancelled) return;

        // Collect unique relations
        const seenEdges = new Set<string>();
        const allRelations: GraphRelation[] = [];
        const entityIdSet = new Set(entities.map((e) => e.entity_id));

        for (const sg of subgraphs) {
          if (!sg) continue;
          // Also add any entities from subgraphs that we haven't seen
          for (const ent of sg.entities) {
            if (!entityIdSet.has(ent.entity_id)) {
              entityIdSet.add(ent.entity_id);
              apiNodes.push(entityToNode(ent, apiNodes.length));
            }
          }
          for (const rel of sg.relations) {
            const key = `${rel.source_id}-${rel.target_id}-${rel.relation_type}`;
            if (!seenEdges.has(key)) {
              seenEdges.add(key);
              allRelations.push(rel);
            }
          }
        }

        // Only keep edges whose source and target exist in our node set
        const finalEntityIds = new Set(apiNodes.map((n) => n.id));
        const validRelations = allRelations.filter(
          (r) => finalEntityIds.has(r.source_id) && finalEntityIds.has(r.target_id),
        );

        const apiEdges = validRelations.map(relationToEdge);

        setNodes(apiNodes);
        setEdges(apiEdges);
        setIsDemo(false);
        setLoading(false);
      } catch {
        // Backend not reachable or error — fall back to demo
        if (cancelled) return;
        setNodes(DEMO_NODES);
        setEdges(DEMO_EDGES);
        setIsDemo(true);
        setLoading(false);
      }
    }

    loadGraph();

    return () => {
      cancelled = true;
    };
  }, [setNodes, setEdges]);

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="border-b border-border/50 px-6 py-4">
        <div className="flex items-center gap-3">
          <div className="flex size-9 items-center justify-center rounded-lg bg-cyan-500/15">
            <Network className="size-5 text-cyan-400" />
          </div>
          <div>
            <h1 className="text-xl font-semibold tracking-tight">Knowledge Graph</h1>
            <p className="text-xs text-muted-foreground">
              {loading
                ? "Loading graph data..."
                : isDemo
                  ? "Demo visualization — connect to the ARA backend for real data"
                  : stats
                    ? `${stats.entities} entities, ${stats.relations} relations`
                    : "Connected to ARA backend"}
            </p>
          </div>
        </div>
      </div>

      {/* Graph */}
      <div className="relative flex-1">
        {loading ? (
          <div className="flex h-full items-center justify-center">
            <div className="flex flex-col items-center gap-3">
              <Loader2 className="size-8 animate-spin text-cyan-400" />
              <p className="text-sm text-muted-foreground">Loading knowledge graph...</p>
            </div>
          </div>
        ) : (
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            nodeTypes={nodeTypes}
            fitView
            proOptions={{ hideAttribution: true }}
            minZoom={0.3}
            maxZoom={2}
            className="bg-background"
          >
            <Background color="#27272a" gap={20} size={1} />
            <Controls
              className="!bg-zinc-900 !border-border/50 !rounded-lg !shadow-lg [&>button]:!bg-zinc-800 [&>button]:!border-border/50 [&>button]:!text-zinc-400 [&>button:hover]:!bg-zinc-700"
            />
          </ReactFlow>
        )}

        {/* Legend */}
        {!loading && (
          <div className="absolute bottom-4 left-4 z-10">
            <Card className="border-border/50 bg-zinc-900/90 backdrop-blur-sm">
              <CardContent className="py-3 px-4">
                <div className="flex flex-wrap items-center gap-2">
                  {Object.entries(TYPE_STYLES).map(([type, style]) => (
                    <Badge
                      key={type}
                      variant="outline"
                      className={`border text-[10px] ${style.badge}`}
                    >
                      {type}
                    </Badge>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Info banner — only show for demo mode */}
        {!loading && isDemo && (
          <div className="absolute top-4 right-4 z-10">
            <Card className="border-border/50 bg-zinc-900/90 backdrop-blur-sm max-w-xs">
              <CardContent className="flex items-start gap-2 py-3 px-4">
                <Info className="size-4 shrink-0 text-cyan-400 mt-0.5" />
                <p className="text-xs text-muted-foreground leading-relaxed">
                  This is a demo visualization. Connect to the ARA backend to explore
                  your personal knowledge graph with real entities and relationships.
                </p>
              </CardContent>
            </Card>
          </div>
        )}
      </div>

      {/* Footer note */}
      <div className="border-t border-border/50 px-6 py-3">
        <p className="text-center text-xs text-muted-foreground/60">
          {isDemo
            ? "Connect to ARA backend to explore your personal knowledge graph"
            : "Showing live data from your ARA knowledge graph"}
        </p>
      </div>
    </div>
  );
}
