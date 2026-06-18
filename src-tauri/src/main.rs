// Project Commentator desktop shell.
//
// Inference runs in-process via llama-cpp-2 (Metal GPU + mtmd vision). On launch
// the single brain — the QAT gemma-4-E2B text model plus its mmproj projector —
// is downloaded (first run) and loaded. A background driver then screenshots the
// primary monitor every ~60s, asks the model for a funny comment, and pushes it
// to the React UI over a Tauri event ("comment"). Status (download/load
// progress) is pushed over the "status" event.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use project_commentator::driver;
use project_commentator::inference::Engine;
use tauri::{Manager, RunEvent};

fn main() {
    let app = tauri::Builder::default()
        .setup(|app| {
            let engine = Engine::new();
            engine.start();
            driver::spawn(engine.clone(), app.handle().clone());
            app.manage(engine);
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building Project Commentator");

    // We intercept quit and terminate the process with the raw POSIX `_exit`
    // instead of letting the Tauri/tao event loop fall through to the normal
    // `std::process::exit`. ggml's Metal backend registers a C++ static
    // destructor that runs via libc `atexit` during `exit()`; on quit that
    // destructor tears down the Metal device and calls `ggml_abort` from inside
    // `ggml_metal_rsets_free`, which `abort()`s the process — that is what macOS
    // surfaces as "Project Commentator closed unexpectedly". `_exit(0)` skips
    // `atexit` entirely, so the destructor never runs and the OS just reclaims
    // the process cleanly. Everything we need to persist is already on disk
    // (the brain is downloaded before any capture), so skipping teardown is safe.
    app.run(|_app_handle, event| {
        if matches!(event, RunEvent::ExitRequested { .. } | RunEvent::Exit) {
            unsafe { libc::_exit(0) };
        }
    });
}