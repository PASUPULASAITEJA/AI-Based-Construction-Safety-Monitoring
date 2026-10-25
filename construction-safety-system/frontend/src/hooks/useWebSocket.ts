import { useState, useEffect, useRef, useCallback } from 'react';
import { TelemetryMessage } from '../types';

export const useWebSocket = (path: string = '/ws/telemetry') => {
  const [telemetry, setTelemetry] = useState<TelemetryMessage | null>(null);
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const socketRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);
  const pingIntervalRef = useRef<number | null>(null);
  const retryCountRef = useRef<number>(0);

  const connect = useCallback(() => {
    // Resolve protocol and host
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const wsUrl = `${protocol}//${host}${path}`;

    try {
      const ws = new WebSocket(wsUrl);
      socketRef.current = ws;

      ws.onopen = () => {
        setIsConnected(true);
        setError(null);
        retryCountRef.current = 0;

        // Start ping heartbeat
        pingIntervalRef.current = window.setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send('ping');
          }
        }, 15000);
      };

      ws.onmessage = (event: MessageEvent) => {
        if (event.data === 'pong') return;
        try {
          const data: TelemetryMessage = JSON.parse(event.data);
          if (data.type === 'FRAME_TELEMETRY') {
            setTelemetry(data);
          }
        } catch (e) {
          console.error('Failed to parse WebSocket JSON payload', e);
        }
      };

      ws.onerror = (e) => {
        setError('WebSocket connection error.');
      };

      ws.onclose = () => {
        setIsConnected(false);
        if (pingIntervalRef.current) clearInterval(pingIntervalRef.current);

        // Exponential backoff reconnect
        const delay = Math.min(1000 * Math.pow(1.5, retryCountRef.current), 10000);
        retryCountRef.current += 1;
        reconnectTimeoutRef.current = window.setTimeout(() => {
          connect();
        }, delay);
      };
    } catch (err: any) {
      setError(err.message);
    }
  }, [path]);

  useEffect(() => {
    connect();

    return () => {
      if (socketRef.current) {
        socketRef.current.close();
      }
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
      if (pingIntervalRef.current) clearInterval(pingIntervalRef.current);
    };
  }, [connect]);

  return { telemetry, isConnected, error };
};
