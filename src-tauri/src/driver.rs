// The only module that touches Tauri. It kicks off the brain load, pumps
// "status" events to the UI until the model is ready, then on a fixed
// interval captures the screen, asks the model for a funny comment, and emits
// a "comment" event with the text + a thumbnail.

use std::time::Duration;

use serde::Serialize;
use tauri::{AppHandle, Emitter};

use crate::capture;
use crate::inference::{self, Engine, Phase};

/// Seconds between captures. Override with COMMENTATOR_INTERVAL_SECS for fast
/// iteration (e.g. `COMMENTATOR_INTERVAL_SECS=5 npm run dev`).
const DEFAULT_INTERVAL_SECS: u64 = 60;

const MAX_TOKENS: usize = 320;

const SYSTEM_PROMPT: &str = "You are a witty, snarky commentator. You are shown a screenshot of \
    someone's screen. Reply with a short, genuinely funny paragraph about whatever is on screen. \
    No preamble, no quotation marks, just the paragraph.";

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CommentPayload {
    pub id: u64,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thumb: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
    pub ts: String,
}

/// Spawn the background driver on the Tauri (tokio) runtime.
pub fn spawn(engine: Engine, app: AppHandle) {
    tauri::async_runtime::spawn(async move {
        // Kick off the download + load.
        if let Err(e) = engine.load(crate::brains::ACTIVE) {
            let _ = app.emit(
                "status",
                inference::Status::error(format!("could not start the engine: {e}")),
            );
            return;
        }

        // Phase 1: pump status to the UI until the model is ready.
        let mut last_phase = None;
        let mut last_progress = -1.0f32;
        loop {
            let s = engine.status();
            let changed = last_phase != Some(s.phase)
                || (s.phase == Phase::Downloading && (s.progress - last_progress).abs() >= 0.01);
            if changed {
                last_phase = Some(s.phase);
                last_progress = s.progress;
                let _ = app.emit("status", s.clone());
            }
            if s.phase == Phase::Ready || s.phase == Phase::Error {
                break;
            }
            tokio::time::sleep(Duration::from_millis(400)).await;
        }

        // If the model failed to load, stop here — the gate is showing the error.
        if engine.status().phase != Phase::Ready {
            return;
        }

        // Phase 2: capture → comment on a fixed interval.
        let secs = std::env::var("COMMENTATOR_INTERVAL_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_INTERVAL_SECS);
        let mut tick = tokio::time::interval(Duration::from_secs(secs.max(1)));
        let mut id = 0u64;
        loop {
            tick.tick().await;
            if let Err(e) = run_once(&engine, &app, &mut id).await {
                eprintln!("[commentator] tick failed: {e}");
            }
        }
    });
}

async fn run_once(engine: &Engine, app: &AppHandle, id: &mut u64) -> Result<(), String> {
    let cap = capture::capture_primary()?;
    let instruction = format!(
        "{} Look at this screenshot and roast it in one funny paragraph.",
        inference::media_marker()
    );
    // Thinking burns tokens on reasoning before the answer — give it room.
    let max_tokens = if inference::thinking_enabled() { 1024 } else { MAX_TOKENS };
    let reply = engine
        .generate_with_image(
            SYSTEM_PROMPT,
            &instruction,
            cap.rgb,
            cap.nx,
            cap.ny,
            max_tokens,
        )
        .await?;

    *id += 1;
    let payload = CommentPayload {
        id: *id,
        text: reply.text,
        thumb: Some(cap.thumb_base64),
        thinking: reply.thinking,
        ts: chrono::Local::now().to_rfc3339(),
    };
    app.emit("comment", payload).map_err(|e| e.to_string())?;
    Ok(())
}