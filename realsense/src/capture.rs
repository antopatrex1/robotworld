use crate::ipc::Frame;
use anyhow::Result;
use image::{DynamicImage, ImageBuffer, ImageFormat, Luma, RgbImage};
use serde_json::{Value, json};
use std::{io::Cursor, path::Path};

pub struct Capture {
    pub metadata: Value,
    pub color_png: Vec<u8>,
    pub depth_preview_png: Vec<u8>,
}

/// Fixed 0.2–4 m scale. Missing depth is black; near is warm, far is cool.
pub fn depth_rgb(frame: &Frame) -> Vec<u8> {
    frame
        .depth
        .iter()
        .flat_map(|&raw| {
            if raw == 0 {
                return [0, 0, 0];
            }
            let t = ((raw as f32 * frame.depth_scale_m - 0.2) / 3.8).clamp(0., 1.);
            let stops = [
                [255., 85., 55.],
                [255., 208., 80.],
                [45., 205., 180.],
                [45., 90., 210.],
            ];
            let x = t * 3.;
            let i = (x as usize).min(2);
            let f = x - i as f32;
            std::array::from_fn(|c| (stops[i][c] * (1. - f) + stops[i + 1][c] * f) as u8)
        })
        .collect()
}

fn png(image: DynamicImage) -> Result<Vec<u8>> {
    let mut bytes = Cursor::new(Vec::new());
    image.write_to(&mut bytes, ImageFormat::Png)?;
    Ok(bytes.into_inner())
}

pub fn save(frame: &Frame, root: &Path) -> Result<Capture> {
    frame.validate()?;
    std::fs::create_dir_all(root)?;
    let root = root.canonicalize()?;
    // Atomic directory creation prevents overwriting simultaneous captures.
    let mut suffix = 0;
    let dir = loop {
        let dir = root.join(format!(
            "{}-{}-{suffix}",
            frame.captured_unix_ms, frame.sequence
        ));
        match std::fs::create_dir(&dir) {
            Ok(()) => break dir,
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => suffix += 1,
            Err(e) => return Err(e.into()),
        }
    };
    let color_png = png(DynamicImage::ImageRgb8(
        RgbImage::from_raw(frame.width, frame.height, frame.color.clone()).unwrap(),
    ))?;
    let depth_preview_png = png(DynamicImage::ImageRgb8(
        RgbImage::from_raw(frame.width, frame.height, depth_rgb(frame)).unwrap(),
    ))?;
    let depth = ImageBuffer::<Luma<u16>, Vec<u16>>::from_raw(
        frame.width,
        frame.height,
        frame.depth.clone(),
    )
    .unwrap();
    let depth_png = png(DynamicImage::ImageLuma16(depth))?;
    std::fs::write(dir.join("color.png"), &color_png)?;
    std::fs::write(dir.join("depth-preview.png"), &depth_preview_png)?;
    std::fs::write(dir.join("depth.png"), &depth_png)?;
    let metadata = json!({
        "directory": dir, "color": dir.join("color.png"),
        "depth": dir.join("depth.png"), "depth_preview": dir.join("depth-preview.png"),
        "width": frame.width, "height": frame.height, "sequence": frame.sequence,
        "captured_unix_ms": frame.captured_unix_ms, "serial": frame.serial, "demo": frame.demo,
        "depth_scale_m": frame.depth_scale_m,
        "depth_encoding": "16-bit unsigned PNG; meters = pixel * depth_scale_m; zero = invalid",
        "depth_preview_range_m": [0.2, 4.0],
        "aligned_to_color": false,
        "color_timestamp_ms": frame.color_timestamp_ms, "depth_timestamp_ms": frame.depth_timestamp_ms,
        "color_frame_number": frame.color_frame_number, "depth_frame_number": frame.depth_frame_number,
        "pairing": "Both images come from one librealsense frameset; native camera coordinates, not pixel-aligned."
    });
    std::fs::write(
        dir.join("metadata.json"),
        serde_json::to_vec_pretty(&metadata)?,
    )?;
    Ok(Capture {
        metadata,
        color_png,
        depth_preview_png,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn raw_depth_roundtrip_and_unique_capture() -> Result<()> {
        let mut frame = crate::camera::demo_frame(1);
        frame.depth[0..4].copy_from_slice(&[0, 1, 1000, 65535]);
        let root = std::env::temp_dir().join(format!(
            "realsense-test-{}-{}",
            std::process::id(),
            crate::ipc::now_ms()
        ));
        let a = save(&frame, &root)?;
        let b = save(&frame, &root)?;
        assert_ne!(a.metadata["directory"], b.metadata["directory"]);
        let loaded = image::open(a.metadata["depth"].as_str().unwrap())?.into_luma16();
        assert_eq!(&loaded.as_raw()[0..4], &[0, 1, 1000, 65535]);
        assert_eq!(&depth_rgb(&frame)[0..3], &[0, 0, 0]);
        std::fs::remove_dir_all(root)?;
        Ok(())
    }
}
