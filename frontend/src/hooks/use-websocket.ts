"use client";

import { useEffect, useRef, useCallback, useState } from "react";
import type { WSMessage, WSResponse } from "@/types/api";

// WebSocket connections cannot be proxied through Next.js rewrites, so we
// connect directly to the backend.  Use NEXT_PUBLIC_WS_URL for explicit
// override, or derive from NEXT_PUBLIC_API_URL, falling back to the default
// backend address.  In production the reverse proxy handles WS natively.
const WS_URL = (() => {
  if (process.env.NEXT_PUBLIC_WS_URL) {
    return process.env.NEXT_PUBLIC_WS_URL;
  }
  const base = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
  return base.replace(/^http/, "ws") + "/ws";
})();
const PING_INTERVAL_MS = 30_000;
const DEFAULT_RECONNECT_MS = 3_000;

// ---------------------------------------------------------------------------
// Options & return types
// ---------------------------------------------------------------------------

interface UseWebSocketOptions {
  onMessage?: (data: WSResponse) => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
  autoConnect?: boolean;
  reconnectInterval?: number;
}

interface UseWebSocketReturn {
  send: (msg: WSMessage) => void;
  sendChat: (content: string, conversationId?: string) => void;
  cancel: () => void;
  isConnected: boolean;
  connect: () => void;
  disconnect: () => void;
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useWebSocket(
  options: UseWebSocketOptions = {},
): UseWebSocketReturn {
  const {
    onMessage,
    onConnect,
    onDisconnect,
    autoConnect = true,
    reconnectInterval = DEFAULT_RECONNECT_MS,
  } = options;

  const wsRef = useRef<WebSocket | null>(null);
  const pingRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const reconnectRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const intentionalCloseRef = useRef(false);

  // Keep callbacks in refs so we don't re-run the effect on every render.
  const onMessageRef = useRef(onMessage);
  const onConnectRef = useRef(onConnect);
  const onDisconnectRef = useRef(onDisconnect);
  onMessageRef.current = onMessage;
  onConnectRef.current = onConnect;
  onDisconnectRef.current = onDisconnect;

  const [isConnected, setIsConnected] = useState(false);

  // -- helpers --------------------------------------------------------------

  const clearTimers = useCallback(() => {
    if (pingRef.current) {
      clearInterval(pingRef.current);
      pingRef.current = null;
    }
    if (reconnectRef.current) {
      clearTimeout(reconnectRef.current);
      reconnectRef.current = null;
    }
  }, []);

  const scheduleReconnect = useCallback(() => {
    if (intentionalCloseRef.current) return;
    reconnectRef.current = setTimeout(() => {
      connect(); // eslint-disable-line @typescript-eslint/no-use-before-define
    }, reconnectInterval);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reconnectInterval]);

  // -- connect / disconnect -------------------------------------------------

  const connect = useCallback(() => {
    // Prevent duplicate connections.
    if (
      wsRef.current &&
      (wsRef.current.readyState === WebSocket.OPEN ||
        wsRef.current.readyState === WebSocket.CONNECTING)
    ) {
      return;
    }

    intentionalCloseRef.current = false;
    clearTimers();

    const ws = new WebSocket(WS_URL);

    ws.onopen = () => {
      setIsConnected(true);
      onConnectRef.current?.();

      // Start keep-alive pings.
      pingRef.current = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(
            JSON.stringify({
              type: "ping",
              content: "",
              conversation_id: "",
              metadata: {},
            } satisfies WSMessage),
          );
        }
      }, PING_INTERVAL_MS);
    };

    ws.onmessage = (event) => {
      try {
        const data: WSResponse = JSON.parse(event.data as string);
        onMessageRef.current?.(data);
      } catch {
        // Ignore malformed frames.
      }
    };

    ws.onclose = () => {
      setIsConnected(false);
      clearTimers();
      onDisconnectRef.current?.();
      scheduleReconnect();
    };

    ws.onerror = () => {
      // `onclose` fires after `onerror`, so reconnect is handled there.
    };

    wsRef.current = ws;
  }, [clearTimers, scheduleReconnect]);

  const disconnect = useCallback(() => {
    intentionalCloseRef.current = true;
    clearTimers();
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setIsConnected(false);
  }, [clearTimers]);

  // -- send helpers ---------------------------------------------------------

  const send = useCallback((msg: WSMessage) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(msg));
    }
  }, []);

  const sendChat = useCallback(
    (content: string, conversationId?: string) => {
      send({
        type: "chat",
        content,
        conversation_id: conversationId ?? "",
        metadata: {},
      });
    },
    [send],
  );

  const cancel = useCallback(() => {
    send({
      type: "cancel",
      content: "",
      conversation_id: "",
      metadata: {},
    });
  }, [send]);

  // -- lifecycle ------------------------------------------------------------

  useEffect(() => {
    if (autoConnect) {
      connect();
    }

    return () => {
      disconnect();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return { send, sendChat, cancel, isConnected, connect, disconnect };
}
