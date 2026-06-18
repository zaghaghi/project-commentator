import { useEffect, useRef, useState } from "react";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import type { ServerStatus } from "./useServerStatus";

export interface Comment {
  id: number;
  text: string;
  thumb?: string;
  thinking?: string;
  ts: string;
}

// Subscribes to the two Tauri events the backend emits: "status" (model
// download/load progress) and "comment" (each roast). Returns the latest
// status and the growing comment list.
//
// `listen` is async, so a naîve cleanup that just calls the unlisten refs can
// race: in React 18 StrictMode (dev) the effect is torn down *before* the first
// `await listen` resolves, the unlisten refs are still undefined, and the
// cleanup no-ops — leaving a listener registered that never gets removed. On
// remount a second listener registers, so every event fires twice (the
// "everything posted twice" symptom). The `cancelled` flag + unlisten-on-
// resolve fixes that race: if cleanup already ran, we unlisten the moment the
// async resolves; otherwise we stash the unlisten fn for the real cleanup.
export function useComments() {
  const [status, setStatus] = useState<ServerStatus | null>(null);
  const [comments, setComments] = useState<Comment[]>([]);
  const idRef = useRef(0);

  useEffect(() => {
    let cancelled = false;
    const unlisteners: UnlistenFn[] = [];
    (async () => {
      const unStatus = await listen<ServerStatus>("status", (e) =>
        setStatus(e.payload),
      );
      if (cancelled) {
        unStatus();
        return;
      }
      unlisteners.push(unStatus);

      const unComment = await listen<Comment>("comment", (e) =>
        setComments((cs) => [...cs, { ...e.payload, id: ++idRef.current }]),
      );
      if (cancelled) {
        unComment();
        return;
      }
      unlisteners.push(unComment);
    })();
    return () => {
      cancelled = true;
      for (const un of unlisteners) un();
    };
  }, []);

  return { status, comments };
}