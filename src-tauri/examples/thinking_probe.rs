// Thinking probe: load the brain, run one image-grounded turn with a synthetic
// image (so no Screen Recording permission is needed), and — with
// COMMENTATOR_DEBUG_RAW=1 — dump the rendered prompt and the raw model output.
// That reveals whether the model's chat template actually injects thinking
// markers and whether the model emits a thought channel we can parse.
//
//   COMMENTATOR_DEBUG_RAW=1 cargo run --release --example thinking_probe -p project-commentator
//
// First run downloads ~4.6 GB into ~/.project-commentator/brains/ (cached after).

use std::time::Duration;

use project_commentator::brains;
use project_commentator::inference::{self, Engine, Phase};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine = Engine::new();
    engine.start();
    engine.load(brains::ACTIVE)?;

    loop {
        let s = engine.status();
        println!(
            "[probe] phase={:?} progress={:.2} msg={} bytes={}/{}",
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

    println!(
        "[probe] thinking_enabled={} (set COMMENTATOR_THINKING=false to disable)",
        inference::thinking_enabled()
    );

    // Sample thinking across several turns with varied synthetic images (no
    // Screen Recording permission needed). Thinking is stochastic at temp 0.8
    // and may depend on image complexity, so one shot isn't conclusive.
    let (nx, ny) = (96u32, 96u32);
    let instruction = format!(
        "{} Look at this screenshot and roast it in one funny sentence.",
        inference::media_marker()
    );
    let mut saw_thinking = 0usize;
    for turn in 0..5u32 {
        // A different pseudo-random-ish pattern each turn (no Math/random in the
        // workflow sense — this is a plain example, SystemTime is fine here).
        let base = ((turn as u32).wrapping_mul(37)) as u8;
        let rgb: Vec<u8> = (0..(nx * ny * 3))
            .map(|i| {
                let v = ((i as u32).wrapping_add(base as u32 * 7)) % 256;
                v as u8
            })
            .collect();
        let reply = engine
            .generate_with_image(
                "You are a witty, snarky commentator. Reply with ONE short, genuinely funny \
                 comment about whatever is on screen. No preamble, no quotation marks, just \
                 the quip.",
                &instruction,
                rgb,
                nx,
                ny,
                1024,
            )
            .await?;
        let has = reply.thinking.is_some();
        if has {
            saw_thinking += 1;
        }
        println!("[probe] turn {turn}: thinking present={has}");
        if let Some(t) = &reply.thinking {
            println!("[probe]   thinking: {t}");
        }
        println!("[probe]   comment: {}", reply.text);
    }
    println!("[probe] thought on {saw_thinking}/5 turns");

    // ggml's Metal atexit destructor aborts on a normal exit (the Tauri shell
    // works around the same bug with libc::_exit). Flush stdout, then exit raw
    // so the probe finishes cleanly instead of reporting a crash after its work
    // is done (block-buffered stdout would otherwise be lost to the abort).
    use std::io::Write;
    let _ = std::io::stdout().flush();
    unsafe { libc::_exit(0) }
}