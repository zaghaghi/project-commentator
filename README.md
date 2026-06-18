# Project Commentator

A desktop app that **watches your screen and roasts it**. Every minute or so it
takes a screenshot of your primary monitor, feeds it to a local Gemma 4 vision
model, and appends the model's roast — a short funny paragraph — to a simple
one-way chat feed (with a thumbnail of the screen it commented on).

Project Commentator is a **standalone macOS desktop app** (Tauri). Everything
runs locally in one process — the model is a quantized Gemma 4 loaded in-process
via llama.cpp on the Metal GPU, with the mmproj vision projector so it can see
the screenshots. **No cloud, no API keys.**

## What it does

On launch the Tauri app:

1. Downloads the brain on first run — the QAT gemma-4-E2B text model **plus**
   its mmproj vision projector from `google/gemma-4-E2B-it-qat-q4_0-gguf` — cached
   in `~/.project-commentator/brains/`. A progress bar shows the download.
2. Loads both on the Metal GPU and inits the mtmd vision context.
3. Every `COMMENTATOR_INTERVAL_SECS` (default 60s): captures the primary monitor,
   asks the model for a funny comment, and emits it to the UI over a Tauri event.

Comments flow one way: the app captures, thinks, and speaks. There is no input
box — it's a feed of roasts.

```
project-commentator/
  src-tauri/
    src/lib.rs         # engine + capture + driver crate
    src/main.rs        # Tauri shell: starts engine + driver, loads the UI
    src/brains.rs      # the brain catalog (QAT main model + mmproj) + HF paths
    src/inference.rs   # llama.cpp engine: download, load, mtmd vision, sampling
    src/capture.rs     # xcap screen capture → RGB + base64 thumbnail
    src/driver.rs       # the loop: capture → generate → emit "comment"; emit "status"
    examples/headless.rs  # no-GUI smoke (exercises the whole pipeline)
  web/                 # React + TypeScript one-way chat UI
```

## The brain

| file | model | size |
|------|-------|------|
| `gemma-4-E2B_q4_0-it.gguf` | QAT text LM | ~3.35 GB |
| `gemma-4-E2B-it-mmproj.gguf` | vision projector | ~1.28 GB |

Both come from the one repo `google/gemma-4-E2B-it-qat-q4_0-gguf`. Edit the
catalog in [brains.rs](src-tauri/src/brains.rs) to change it.

## Prereqs

Rust (stable), Node 20+, and **cmake** + Xcode Command Line Tools (to compile
llama.cpp; full Xcode is *not* required — its Metal shaders compile at runtime).

## Develop

```bash
npm install            # one-time: Tauri CLI at the repo root
npm --prefix web install
npm run dev            # = tauri dev: builds llama.cpp (first time is slow), launches the window
```

Grant **Screen Recording** (System Settings → Privacy & Security → Screen
Recording → enable "Project Commentator"), then quit & `npm run dev` again —
without it, macOS returns a black frame and the app reports the permission error.

Fast iteration:

```bash
COMMENTATOR_INTERVAL_SECS=5 npm run dev   # a capture+comment every 5s
COMMENTATOR_THINKING=true npm run dev      # let the model reason before answering (see below)
```

### Thinking mode

Gemma 4 reasons before answering, and the reasoning is surfaced in the UI. By
default thinking is **on**: the engine renders the model's baked chat template
with `enable_thinking=true` (the model emits reasoning in a thought channel, then
the answer). Each comment shows only the quip; a **"view thinking"** button on
the comment reveals the reasoning, and **"view screenshot"** reveals the screen
it was commenting on. The button only appears on turns where the model actually
reasoned — at temp 0.8 the E2B model sometimes just answers directly, so a given
comment may have no thinking to show (and no button).

Opt out with `COMMENTATOR_THINKING=false` (or `0`) to use the manual Gemma prompt
with no reasoning and a smaller token budget. If the template render fails for
any reason it falls back to the manual prompt so the app keeps working.

The thought/answer split is best-effort and tolerant to token-spelling
differences across templates/quantizations; if you ever see reasoning leaking
into the quip (or the quip empty), tune
[`split_thinking`](src-tauri/src/inference.rs) against the real output (the
headless example prints the raw thinking). Note: on some quants thinking can
trigger an `<unused49>` token-flooding bug — if that happens, set
`COMMENTATOR_THINKING=false`.

## Headless / no-GUI smoke

Bypasses the GUI and exercises the whole pipeline — load, capture, generate,
print:

```bash
cargo run --release --example headless -p project-commentator
```

## Build a distributable

```bash
npm run build      # builds web/dist, compiles the app
```

Output: `src-tauri/target/release/bundle/macos/Project Commentator.app` (small —
**no model inside**; the brain is downloaded on first run).

## Notes

- Each turn starts from a clean KV cache (no cross-turn prompt caching) since
  every screenshot differs.
- Gemma vision needs a non-causal attention mask; it is gated on what the mmproj
  projector reports (`decode_use_non_causal()`), and logged at startup.