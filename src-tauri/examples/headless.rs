// Headless smoke: bypasses the GUI and exercises the whole pipeline — load the
// brain, capture the primary screen, ask the model for a roast, print it.
//
//   cargo run --release --example headless -p project-commentator
//   COMMENTATOR_INTERVAL_SECS=5 cargo run --release --example headless -p project-commentator
//
// First run downloads ~4.6 GB into ~/.project-commentator/brains/. Needs macOS
// Screen Recording permission (else the capture is a blank frame and this
// reports the permission error).

use std::time::Duration;

use project_commentator::brains;
use project_commentator::capture;
use project_commentator::inference::{self, Engine, Phase};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine = Engine::new();
    engine.start();
    engine.load(brains::ACTIVE)?;

    // Wait until the model is ready (or fails).
    loop {
        let s = engine.status();
        println!(
            "[headless] phase={:?} progress={:.2} msg={} bytes={}/{}",
            s.phase, s.progress, s.message, s.downloaded_bytes, s.total_bytes
        );
        match s.phase {
            Phase::Ready => break,
            Phase::Error => {
                return Err(format!("load failed: {}", s.error.unwrap_or_default()).into());
            }
            _ => {}
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }

    let cap = capture::capture_primary()?;
    println!("[headless] captured {}x{}", cap.nx, cap.ny);

    let instruction = format!(
        "{} Look at this screenshot and roast it in one funny sentence.",
        inference::media_marker()
    );
    let reply = engine
        .generate_with_image(
            "You are a witty, snarky commentator. Reply with ONE short, genuinely funny comment \
             about whatever is on screen. No preamble, no quotation marks, just the quip.",
            &instruction,
            cap.rgb,
            cap.nx,
            cap.ny,
            200,
        )
        .await?;
    if let Some(t) = &reply.thinking {
        println!("[headless] thinking: {t}");
    }
    println!("[headless] commentator says: {}", reply.text);

    // ggml's Metal atexit destructor aborts on a normal exit (the Tauri shell
    // works around the same bug with libc::_exit). Flush stdout, then exit raw
    // so the smoke test finishes cleanly instead of reporting a crash.
    use std::io::Write;
    let _ = std::io::stdout().flush();
    unsafe { libc::_exit(0) }
}