// Screen capture via xcap. We grab the primary monitor, pull the raw RGBA
// bytes out of the capture, and rebuild them with our own `image` crate — that
// way we never depend on which exact `image` version xcap links against. The
// full-res RGB stays in Rust and goes to the model; only a small PNG thumbnail
// is base64'd for the UI.

use std::io::Cursor;

use base64::Engine;
use image::imageops::{self, FilterType};
use image::{DynamicImage, ImageFormat, RgbaImage};
use xcap::Monitor;

pub struct Capture {
    /// Full-resolution RGB (nx*ny*3 bytes) — fed to the model.
    pub rgb: Vec<u8>,
    pub nx: u32,
    pub ny: u32,
    /// Small PNG thumbnail, base64-encoded, for the chat feed.
    pub thumb_base64: String,
}

/// Capture the primary monitor (falling back to the first one). Errors out
/// with a clear message when the Screen Recording permission is missing — on
/// macOS that surfaces as a blank frame rather than a capture error.
pub fn capture_primary() -> Result<Capture, String> {
    let monitors = Monitor::all().map_err(|e| format!("list monitors: {e}"))?;
    let monitor = monitors
        .iter()
        .find(|m| m.is_primary().unwrap_or(false))
        .or_else(|| monitors.first())
        .ok_or_else(|| "no monitors found".to_string())?;

    let captured = monitor
        .capture_image()
        .map_err(|e| format!("capture: {e}"))?;

    // Rebuild with our image crate so resize/PNG-encode use one set of types.
    let (nx, ny) = (captured.width(), captured.height());
    let rgba: RgbaImage = RgbaImage::from_raw(nx, ny, captured.as_raw().clone())
        .ok_or_else(|| "capture produced an image whose buffer did not match its dimensions".to_string())?;

    if is_blank_frame(&rgba) {
        return Err(
            "Screen Recording permission not granted. Open System Settings → Privacy & Security \
             → Screen Recording, enable Project Commentator, then restart the app."
                .to_string(),
        );
    }

    let rgb: Vec<u8> = rgba
        .as_raw()
        .chunks(4)
        .flat_map(|c| [c[0], c[1], c[2]])
        .collect();

    let thumb_w = 360u32;
    let thumb_h = ((ny as u64 * thumb_w as u64 / nx.max(1) as u64) as u32).max(1);
    let small = imageops::resize(&rgba, thumb_w, thumb_h, FilterType::Triangle);
    let mut png: Vec<u8> = Vec::new();
    DynamicImage::ImageRgba8(small)
        .write_to(&mut Cursor::new(&mut png), ImageFormat::Png)
        .map_err(|e| format!("encode thumbnail: {e}"))?;
    let thumb_base64 = base64::engine::general_purpose::STANDARD.encode(&png);

    Ok(Capture { rgb, nx, ny, thumb_base64 })
}

/// Heuristic: sample an 8×8 grid; if essentially every sampled pixel is
/// near-black, treat the frame as blank (the macOS no-permission signature).
/// False positive on a genuinely all-black screen is acceptable for v1.
fn is_blank_frame(rgba: &RgbaImage) -> bool {
    let (w, h) = rgba.dimensions();
    if w == 0 || h == 0 {
        return true;
    }
    const GRID: u32 = 8;
    let mut sampled = 0u32;
    let mut dark = 0u32;
    for gy in 0..GRID {
        for gx in 0..GRID {
            let x = ((gx as u64 * w as u64 / (GRID - 1) as u64) as u32).min(w - 1);
            let y = ((gy as u64 * h as u64 / (GRID - 1) as u64) as u32).min(h - 1);
            let p = rgba.get_pixel(x, y);
            sampled += 1;
            if p[0] < 8 && p[1] < 8 && p[2] < 8 {
                dark += 1;
            }
        }
    }
    sampled > 0 && dark * 100 >= sampled * 99
}