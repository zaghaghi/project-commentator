//! Project Commentator inference + capture library: the in-process llama.cpp
//! engine (with mtmd vision), the brain catalog/downloader, and the screen
//! capture + driver that feeds comments to the UI over Tauri events. Shared by
//! the Tauri binary and the headless example.

pub mod brains;
pub mod capture;
pub mod driver;
pub mod inference;