use crate::ipc::{self, Frame, Request, Response};
use anyhow::{Context as _, Result, ensure};
use realsense_rust::{
    config::Config,
    context::Context,
    frame::{ColorFrame, DepthFrame, FrameEx, PixelKind},
    kind::{Rs2CameraInfo, Rs2Format, Rs2StreamKind},
    pipeline::InactivePipeline,
};
use std::{
    os::unix::{
        fs::{FileTypeExt, MetadataExt, PermissionsExt},
        net::{UnixListener, UnixStream},
    },
    path::Path,
    sync::{Arc, Condvar, Mutex},
    time::{Duration, Instant},
};

#[derive(Default)]
struct State {
    frame: Option<Frame>,
    status: String,
    received: Option<Instant>,
}
type Shared = Arc<(Mutex<State>, Condvar)>;

pub fn serve(path: &Path, demo: bool) -> Result<()> {
    let parent = path.parent().context("Socket needs a parent directory")?;
    ensure!(
        parent.is_dir(),
        "Create {} as your normal user first (scripts/start.sh does this)",
        parent.display()
    );
    if path.exists() {
        ensure!(
            std::fs::symlink_metadata(path)?.file_type().is_socket(),
            "Socket path is not a socket"
        );
        ensure!(
            UnixStream::connect(path).is_err(),
            "A camera helper is already running"
        );
        std::fs::remove_file(path)?;
    }
    let listener = UnixListener::bind(path)?;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
    // Root helper exposes only read-only frames to the owner of the runtime directory.
    if unsafe { libc::geteuid() } == 0 {
        use std::os::unix::ffi::OsStrExt;
        let owner = std::fs::metadata(parent)?;
        let cpath = std::ffi::CString::new(path.as_os_str().as_bytes())?;
        ensure!(
            unsafe { libc::chown(cpath.as_ptr(), owner.uid(), owner.gid()) } == 0,
            "Could not set socket owner"
        );
    }
    let shared: Shared = Arc::new((
        Mutex::new(State {
            status: "Starting camera…".into(),
            ..Default::default()
        }),
        Condvar::new(),
    ));
    let camera = shared.clone();
    std::thread::spawn(move || {
        if demo {
            let mut sequence = 0;
            loop {
                sequence += 1;
                publish(&camera, demo_frame(sequence));
                std::thread::sleep(Duration::from_millis(33));
            }
        }
        loop {
            if let Err(error) = stream_camera(&camera) {
                let hint = if unsafe { libc::geteuid() } == 0 {
                    "Helper has administrator access; check the camera log for the underlying error."
                } else {
                    "macOS USB access requires sudo; start scripts/start.sh in Terminal."
                };
                let status = format!("Camera unavailable: {error:#}. {hint}");
                eprintln!("{status}");
                let mut state = camera.0.lock().unwrap();
                state.frame = None;
                state.status = status;
                camera.1.notify_all();
            }
            std::thread::sleep(Duration::from_secs(3));
        }
    });
    eprintln!(
        "Camera helper listening on {}{}",
        path.display(),
        if demo { " [DEMO]" } else { "" }
    );
    // Bounded workers, so idle clients cannot create unbounded threads.
    let listener = Arc::new(listener);
    let mut workers = Vec::new();
    for _ in 0..4 {
        let listener = listener.clone();
        let shared = shared.clone();
        workers.push(std::thread::spawn(move || {
            for mut stream in listener.incoming().flatten() {
                if let Err(e) = respond(&mut stream, &shared) {
                    eprintln!("Camera client: {e}");
                }
            }
        }));
    }
    for worker in workers {
        let _ = worker.join();
    }
    Ok(())
}

fn respond(stream: &mut UnixStream, shared: &Shared) -> Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(3)))?;
    stream.set_write_timeout(Some(Duration::from_secs(3)))?;
    let request: Request = ipc::read_message(stream)?;
    let mut state = shared.0.lock().unwrap();
    if matches!(request, Request::Fresh) {
        let previous = state
            .frame
            .as_ref()
            .map(|f| (f.captured_unix_ms, f.sequence));
        state = shared
            .1
            .wait_timeout_while(state, Duration::from_secs(5), |state| {
                state
                    .frame
                    .as_ref()
                    .map(|f| (f.captured_unix_ms, f.sequence))
                    == previous
            })
            .unwrap()
            .0;
        if state
            .frame
            .as_ref()
            .map(|f| (f.captured_unix_ms, f.sequence))
            == previous
        {
            return ipc::write_message(
                stream,
                &Response {
                    status: format!("Timed out waiting for a fresh frame. {}", state.status),
                    frame: None,
                },
            );
        }
    }
    let fresh = state
        .received
        .is_some_and(|time| time.elapsed() < Duration::from_secs(2));
    let response = Response {
        status: if !fresh && state.frame.is_some() {
            "Camera stalled; waiting for new frames".into()
        } else {
            state.status.clone()
        },
        frame: if fresh && !matches!(request, Request::Status) {
            state.frame.clone()
        } else {
            None
        },
    };
    drop(state);
    ipc::write_message(stream, &response)
}

fn publish(shared: &Shared, frame: Frame) {
    let mut state = shared.0.lock().unwrap();
    state.status = format!(
        "{} · {} × {} · RGB8 + Z16 · frame {}",
        if frame.demo {
            "DEMO — synthetic frames"
        } else {
            "D435 streaming"
        },
        frame.width,
        frame.height,
        frame.sequence
    );
    state.received = Some(Instant::now());
    state.frame = Some(frame);
    shared.1.notify_all();
}

fn stream_camera(shared: &Shared) -> Result<()> {
    let context = Context::new()?;
    let devices = context.query_devices(Default::default());
    let device = devices
        .iter()
        .find(|d| {
            d.info(Rs2CameraInfo::Name)
                .is_some_and(|s| s.to_string_lossy().contains("D435"))
        })
        .context("No accessible D435 found")?;
    let serial = device
        .info(Rs2CameraInfo::SerialNumber)
        .context("No serial number")?;
    let mut config = Config::new();
    config
        .enable_device_from_serial(serial)?
        .disable_all_streams()?
        .enable_stream(Rs2StreamKind::Color, None, 640, 480, Rs2Format::Rgb8, 30)?
        .enable_stream(Rs2StreamKind::Depth, None, 640, 480, Rs2Format::Z16, 30)?;
    let mut pipeline = InactivePipeline::try_from(&context)?.start(Some(config))?;
    let mut sequence = 0;
    let result = (|| -> Result<()> {
        loop {
            let frames = pipeline.wait(Some(Duration::from_secs(5)))?;
            let Some(color) = frames.frames_of_type::<ColorFrame>().pop() else {
                continue;
            };
            let Some(depth) = frames.frames_of_type::<DepthFrame>().pop() else {
                continue;
            };
            ensure!(
                color.width() == depth.width() && color.height() == depth.height(),
                "Unexpected stream dimensions"
            );
            sequence += 1;
            let mut rgb = Vec::with_capacity(color.width() * color.height() * 3);
            for pixel in color.iter() {
                rgb.extend_from_slice(&rgb_pixel(pixel)?);
            }
            let z = depth
                .iter()
                .filter_map(|pixel| {
                    if let PixelKind::Z16 { depth } = pixel {
                        Some(*depth)
                    } else {
                        None
                    }
                })
                .collect();
            let frame = Frame {
                sequence,
                captured_unix_ms: ipc::now_ms(),
                width: color.width() as u32,
                height: color.height() as u32,
                color: rgb,
                depth: z,
                depth_scale_m: depth.depth_units()?,
                color_timestamp_ms: color.timestamp(),
                depth_timestamp_ms: depth.timestamp(),
                color_frame_number: color.frame_number(),
                depth_frame_number: depth.frame_number(),
                serial: serial.to_string_lossy().into_owned(),
                demo: false,
            };
            frame.validate()?;
            publish(shared, frame);
        }
    })();
    // Stop USB streams explicitly before retrying on a frame/conversion error.
    drop(pipeline.stop());
    result
}

fn rgb_pixel(pixel: PixelKind<'_>) -> Result<[u8; 3]> {
    match pixel {
        // realsense-rust 1.3.0 emits Bgr8 for RGB8 input, but the named channel
        // references are correct. Accept either variant without swapping r/b.
        PixelKind::Rgb8 { r, g, b } | PixelKind::Bgr8 { r, g, b } => Ok([*r, *g, *b]),
        other => anyhow::bail!("Unexpected color pixel format: {other:?}"),
    }
}

pub fn demo_frame(sequence: u64) -> Frame {
    let (width, height) = (640, 480);
    let mut color = Vec::with_capacity(width * height * 3);
    let mut depth = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            let circle = (x as i32 - (sequence % 400 + 120) as i32).pow(2)
                + (y as i32 - 240).pow(2)
                < 85 * 85;
            let pixel = if circle {
                [255, 190, 70]
            } else {
                [((x * 255) / width) as u8, ((y * 255) / height) as u8, 125]
            };
            color.extend_from_slice(&pixel);
            depth.push(if circle { 650 } else { (600 + x * 5) as u16 });
        }
    }
    Frame {
        sequence,
        captured_unix_ms: ipc::now_ms(),
        width: width as u32,
        height: height as u32,
        color,
        depth,
        depth_scale_m: 0.001,
        color_timestamp_ms: sequence as f64 * 33.333,
        depth_timestamp_ms: sequence as f64 * 33.333,
        color_frame_number: sequence,
        depth_frame_number: sequence,
        serial: "SYNTHETIC-DEMO".into(),
        demo: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rgb_binding_variant_preserves_channels() -> Result<()> {
        let (r, g, b) = (11, 72, 203);
        for pixel in [
            PixelKind::Rgb8 {
                r: &r,
                g: &g,
                b: &b,
            },
            PixelKind::Bgr8 {
                r: &r,
                g: &g,
                b: &b,
            },
        ] {
            assert_eq!(rgb_pixel(pixel)?, [11, 72, 203]);
        }
        Ok(())
    }
}
