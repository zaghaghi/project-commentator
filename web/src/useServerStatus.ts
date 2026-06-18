// The status shape the Rust backend emits over the "status" Tauri event. Kept
// here as a type only — the listening lives in useComments.ts. Mirrors
// `inference::Status` (serde camelCase).
export type Phase = "idle" | "downloading" | "loading" | "ready" | "error";

export interface ServerStatus {
  phase: Phase;
  modelReady: boolean;
  modelName: string;
  progress: number;
  downloadedBytes: number;
  totalBytes: number;
  message: string;
  error?: string;
}