import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import type { EngineEvent, LogLine, MissionMeta, MissionState, MissionStep, MissionSummary, Snapshot } from "./types";

export type Connection = "connecting" | "online" | "offline";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

async function request<T>(method: "GET" | "POST", path: string, body?: unknown): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`api/${path}`, {
      method,
      headers: body === undefined ? undefined : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    throw new ApiError(0, "Engine unreachable: is `python -m aegis.console` running?");
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const j = await res.json();
      detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, detail || `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  get: <T,>(path: string) => request<T>("GET", path),
  post: <T,>(path: string, body: unknown = {}) => request<T>("POST", path, body),
};

type Listener = (ev: EngineEvent) => void;

export interface Toast {
  id: number;
  level: "error" | "warn" | "ok";
  text: string;
}

interface EngineContext {
  connection: Connection;
  snapshot: Snapshot | null;
  logs: LogLine[];
  boot: { stage: string; progress: number };
  mission: MissionState | null;
  busy: string | null;
  toasts: Toast[];
  dismissToast: (id: number) => void;
  /** Runs a command; returns null (and raises a toast) on failure. Only one command at a time. */
  run: <T>(name: string, path: string, body?: unknown) => Promise<T | null>;
  subscribe: (type: string, fn: Listener) => () => void;
}

const Ctx = createContext<EngineContext | null>(null);
const MAX_LOGS = 500;

export function EngineProvider({ children }: { children: ReactNode }) {
  const [connection, setConnection] = useState<Connection>("connecting");
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [logs, setLogs] = useState<LogLine[]>([]);
  const [boot, setBoot] = useState({ stage: "", progress: 0 });
  const [mission, setMission] = useState<MissionState | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const listeners = useRef(new Map<string, Set<Listener>>());
  const busyRef = useRef<string | null>(null);
  const toastId = useRef(0);

  const toast = useCallback((level: Toast["level"], text: string) => {
    const id = ++toastId.current;
    setToasts((t) => [...t.slice(-3), { id, level, text }]);
    window.setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), level === "error" ? 7000 : 4000);
  }, []);
  const dismissToast = useCallback((id: number) => setToasts((t) => t.filter((x) => x.id !== id)), []);

  const subscribe = useCallback((type: string, fn: Listener) => {
    const map = listeners.current;
    if (!map.has(type)) map.set(type, new Set());
    map.get(type)!.add(fn);
    return () => {
      map.get(type)?.delete(fn);
    };
  }, []);

  const refreshMission = useCallback(async () => {
    try {
      const m = await api.get<MissionState | Record<string, never>>("mission");
      setMission("id" in m ? (m as MissionState) : null);
    } catch {
      /* engine not ready: no mission */
    }
  }, []);

  // One WebSocket for the whole console, reconnecting with backoff
  useEffect(() => {
    let socket: WebSocket | null = null;
    let closed = false;
    let retry = 0;
    let timer = 0;

    const dispatch = (ev: EngineEvent) => {
      switch (ev.type) {
        case "hello": {
          const d = ev.data as { state: Snapshot; logs: EngineEvent<LogLine>[] };
          setSnapshot(d.state);
          setLogs(d.logs.map((l) => l.data).slice(-MAX_LOGS));
          void refreshMission();
          break;
        }
        case "state":
          setSnapshot(ev.data as Snapshot);
          break;
        case "log":
          setLogs((l) => [...l.slice(-(MAX_LOGS - 1)), ev.data as LogLine]);
          break;
        case "boot": {
          const d = ev.data as { stage: string; progress: number };
          setBoot(d);
          if (d.stage !== "ready") setSnapshot((s) => (s ? { ...s, status: d.stage === "error" ? "error" : "booting", boot_stage: d.stage } : s));
          break;
        }
        case "mission.start":
          setMission({ ...(ev.data as MissionMeta), steps: [], running: true });
          break;
        case "mission.step": {
          const step = ev.data as MissionStep;
          // A resync (GET /mission after reconnect) may already contain this step
          setMission((m) => (m && !m.steps.some((s) => s.step === step.step) ? { ...m, steps: [...m.steps, step] } : m));
          break;
        }
        case "mission.end":
          setMission((m) => (m ? { ...m, running: false, summary: ev.data as MissionSummary } : m));
          break;
        case "mission.error":
          setMission((m) => (m ? { ...m, running: false, error: (ev.data as { error: string }).error } : m));
          break;
      }
      listeners.current.get(ev.type)?.forEach((fn) => fn(ev));
      listeners.current.get("*")?.forEach((fn) => fn(ev));
    };

    const connect = () => {
      if (closed) return;
      setConnection((c) => (c === "online" ? "connecting" : c));
      const url = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}${location.pathname.replace(/[^/]*$/, "")}api/ws`;
      socket = new WebSocket(url);
      socket.onopen = () => {
        retry = 0;
        setConnection("online");
      };
      socket.onmessage = (msg) => {
        try {
          dispatch(JSON.parse(msg.data) as EngineEvent);
        } catch {
          /* malformed frame: ignore */
        }
      };
      socket.onclose = () => {
        if (closed) return;
        setConnection("offline");
        timer = window.setTimeout(connect, Math.min(4000, 500 * 2 ** retry++));
      };
      socket.onerror = () => socket?.close();
    };
    connect();
    return () => {
      closed = true;
      window.clearTimeout(timer);
      socket?.close();
    };
  }, [refreshMission]);

  const run = useCallback(
    async <T,>(name: string, path: string, body?: unknown): Promise<T | null> => {
      if (busyRef.current) {
        toast("warn", `Wait for "${busyRef.current}" to finish`);
        return null;
      }
      busyRef.current = name;
      setBusy(name);
      try {
        return await api.post<T>(path, body ?? {});
      } catch (e) {
        const err = e as ApiError;
        toast(err.status === 409 ? "warn" : "error", err.message);
        return null;
      } finally {
        busyRef.current = null;
        setBusy(null);
      }
    },
    [toast],
  );

  const value = useMemo(
    () => ({ connection, snapshot, logs, boot, mission, busy, toasts, dismissToast, run, subscribe }),
    [connection, snapshot, logs, boot, mission, busy, toasts, dismissToast, run, subscribe],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useEngine() {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useEngine outside EngineProvider");
  return ctx;
}

/** Subscribes to one engine event type for the lifetime of the component. */
export function useEngineEvent<T>(type: string, fn: (data: T, ev: EngineEvent<T>) => void) {
  const { subscribe } = useEngine();
  const ref = useRef(fn);
  ref.current = fn;
  useEffect(() => subscribe(type, (ev) => ref.current(ev.data as T, ev as EngineEvent<T>)), [subscribe, type]);
}

/** Absolute URL of a chip thumbnail served by the engine. */
export const chipUrl = (index: number, scale = 1) => `api/chip/${index}.png?scale=${scale}`;
