"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  MessageSquare,
  Brain,
  Network,
  Cpu,
  Settings,
  Upload,
  Wifi,
  WifiOff,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useEffect, useState } from "react";

const navItems = [
  { href: "/", label: "Chat", icon: MessageSquare },
  { href: "/memory", label: "Memory", icon: Brain },
  { href: "/memory/upload", label: "Upload Diaries", icon: Upload },
  { href: "/graph", label: "Knowledge Graph", icon: Network },
  { href: "/models", label: "Models", icon: Cpu },
  { href: "/settings", label: "Settings", icon: Settings },
];

export function SidebarNav() {
  const pathname = usePathname();
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    const checkConnection = async () => {
      try {
        const res = await fetch("http://localhost:8000/health", {
          method: "GET",
          signal: AbortSignal.timeout(3000),
        });
        setConnected(res.ok);
      } catch {
        setConnected(false);
      }
    };

    checkConnection();
    const interval = setInterval(checkConnection, 10000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex h-full flex-col">
      {/* Logo */}
      <div className="flex h-16 items-center gap-2 px-6">
        <span className="text-xl font-bold tracking-wider text-foreground drop-shadow-[0_0_8px_rgba(99,102,241,0.6)]">
          ARA
        </span>
        <span className="text-xs text-muted-foreground">v0.1</span>
      </div>

      {/* Navigation */}
      <nav className="flex-1 space-y-1 px-3 py-4">
        {navItems.map((item) => {
          const isActive =
            item.href === "/"
              ? pathname === "/"
              : pathname.startsWith(item.href);

          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "flex items-center gap-3 rounded-md px-3 py-2.5 text-sm font-medium transition-colors",
                isActive
                  ? "bg-accent text-accent-foreground shadow-[inset_2px_0_0_0_hsl(var(--chart-1))]"
                  : "text-muted-foreground hover:bg-accent/50 hover:text-foreground"
              )}
            >
              <item.icon className={cn("size-4 shrink-0", isActive && "text-[hsl(var(--chart-1))]")} />
              {item.label}
            </Link>
          );
        })}
      </nav>

      {/* Connection Status */}
      <div className="border-t border-border px-4 py-4">
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          {connected ? (
            <>
              <Wifi className="size-3.5 text-emerald-500" />
              <span className="flex items-center gap-1.5">
                <span className="relative flex size-1.5">
                  <span className="absolute inline-flex size-full animate-ping rounded-full bg-emerald-400 opacity-75" />
                  <span className="relative inline-flex size-1.5 rounded-full bg-emerald-500" />
                </span>
                Connected
              </span>
            </>
          ) : (
            <>
              <WifiOff className="size-3.5 text-red-500" />
              <span className="flex items-center gap-1.5">
                <span className="size-1.5 rounded-full bg-red-500" />
                Disconnected
              </span>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export function Sidebar() {
  return (
    <aside className="hidden w-[260px] shrink-0 border-r border-border bg-card md:block">
      <SidebarNav />
    </aside>
  );
}
