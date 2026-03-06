"use client";

import { useCallback, useMemo } from "react";
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
import { Network, Info } from "lucide-react";

import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

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
  return (
    TYPE_STYLES[type] ?? {
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
// Demo data
// ---------------------------------------------------------------------------

const INITIAL_NODES: Node<GraphNodeData>[] = [
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

const INITIAL_EDGES: Edge[] = [
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
// Main page
// ---------------------------------------------------------------------------

export default function GraphPage() {
  const [nodes, , onNodesChange] = useNodesState(INITIAL_NODES);
  const [edges, , onEdgesChange] = useEdgesState(INITIAL_EDGES);

  const nodeTypes = useMemo(() => ({ graphNode: GraphNode }), []);

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
              Graph visualization is available when connected to the ARA backend
            </p>
          </div>
        </div>
      </div>

      {/* Graph */}
      <div className="relative flex-1">
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

        {/* Legend */}
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

        {/* Info banner */}
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
      </div>

      {/* Footer note */}
      <div className="border-t border-border/50 px-6 py-3">
        <p className="text-center text-xs text-muted-foreground/60">
          Connect to ARA backend to explore your personal knowledge graph
        </p>
      </div>
    </div>
  );
}
