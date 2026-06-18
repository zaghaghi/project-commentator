import type { ServerStatus } from "../useServerStatus";

function fmtBytes(n: number): string {
  if (!n) return "0 B";
  const u = ["B", "KB", "MB", "GB"];
  const i = Math.min(u.length - 1, Math.floor(Math.log(n) / Math.log(1024)));
  return `${(n / 1024 ** i).toFixed(i ? 1 : 0)} ${u[i]}`;
}

export function StartupGate({ status }: { status: ServerStatus | null }) {
  // null => no status event yet (backend still booting)
  const phase = status?.phase ?? "connecting";
  const pct = status ? Math.round(status.progress * 100) : 0;

  const heading: Record<string, string> = {
    connecting: "Waking the commentator…",
    idle: "Awaiting the brain…",
    downloading: "Downloading the brain…",
    loading: "Spinning up the cortex…",
    error: "Something went wrong",
    ready: "Ready",
  };

  const showBar = phase === "downloading";

  return (
    <div className="setup">
      <div className="panel startup">
        <div className="dossier">PROJECT COMMENTATOR // BOOTING UP</div>
        <h2>{heading[phase] ?? "Preparing…"}</h2>

        {showBar && (
          <>
            <div className="bar">
              <div className="bar-fill" style={{ width: `${pct}%` }} />
            </div>
            <div className="bar-meta">
              <span>{pct}%</span>
              <span>
                {fmtBytes(status!.downloadedBytes)} / {fmtBytes(status!.totalBytes)}
              </span>
            </div>
            <p className="startup-note">
              First run only — the model is cached for next time.
            </p>
          </>
        )}

        {!showBar && phase !== "error" && (
          <div className="spinner-row">
            <span className="spinner" />
            <span>{status?.message ?? "Waiting for the local model…"}</span>
          </div>
        )}

        {phase === "error" && (
          <div className="error" style={{ marginTop: 12 }}>
            {status?.error ?? "The model could not be prepared."}
          </div>
        )}
      </div>
    </div>
  );
}