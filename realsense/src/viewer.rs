use crate::{capture, ipc};
use gpui::{
    App, Application, Bounds, Context, RenderImage, Window, WindowBounds, WindowOptions, div, img,
    prelude::*, px, rgb, size,
};
use std::{
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

struct Feed {
    status: String,
    frame: Option<ipc::Frame>,
}

struct Viewer {
    feed: Arc<Mutex<Feed>>,
    stop: Arc<AtomicBool>,
    socket: PathBuf,
    output: PathBuf,
    sequence: Option<(u64, u64)>,
    color: Option<Arc<RenderImage>>,
    depth: Option<Arc<RenderImage>>,
    status: String,
    detail: String,
    capture_status: Arc<Mutex<String>>,
    capturing: Arc<AtomicBool>,
}

impl Drop for Viewer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

fn render_image(width: u32, height: u32, rgb: &[u8]) -> Arc<RenderImage> {
    // GPUI's renderer expects BGRA, although image::Frame names its buffer RGBA.
    let bgra = rgb
        .as_chunks::<3>()
        .0
        .iter()
        .flat_map(|p| [p[2], p[1], p[0], 255])
        .collect();
    let buffer = image::RgbaImage::from_raw(width, height, bgra).unwrap();
    Arc::new(RenderImage::new(smallvec::smallvec![image::Frame::new(
        buffer
    )]))
}

impl Viewer {
    fn new(socket: PathBuf, output: PathBuf, cx: &mut Context<Self>) -> Self {
        let feed = Arc::new(Mutex::new(Feed {
            status: "Connecting to camera helper…".into(),
            frame: None,
        }));
        let stop = Arc::new(AtomicBool::new(false));
        let worker_feed = feed.clone();
        let worker_stop = stop.clone();
        let worker_socket = socket.clone();
        std::thread::spawn(move || {
            while !worker_stop.load(Ordering::Relaxed) {
                let response = ipc::request(&worker_socket, ipc::Request::Latest);
                let mut feed = worker_feed.lock().unwrap();
                match response {
                    Ok(response) => {
                        feed.status = response.status;
                        feed.frame = response.frame;
                    }
                    Err(error) => {
                        feed.status = format!("{error:#}");
                        feed.frame = None;
                    }
                }
                drop(feed);
                std::thread::sleep(Duration::from_millis(33));
            }
        });
        cx.spawn(async move |this, cx| {
            loop {
                cx.background_executor()
                    .timer(Duration::from_millis(33))
                    .await;
                if this.update(cx, |_, cx| cx.notify()).is_err() {
                    break;
                }
            }
        })
        .detach();
        Self {
            feed,
            stop,
            socket,
            output,
            sequence: None,
            color: None,
            depth: None,
            status: String::new(),
            detail: "640 × 480 / 30 fps requested".into(),
            capture_status: Arc::new(Mutex::new("Captures include raw metric depth + RGB".into())),
            capturing: Arc::new(AtomicBool::new(false)),
        }
    }

    fn update_images(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let feed = self.feed.lock().unwrap();
        self.status = feed.status.clone();
        if let Some(frame) = &feed.frame {
            let sequence = (frame.captured_unix_ms, frame.sequence);
            if self.sequence != Some(sequence) {
                if let Some(image) = self.color.take() {
                    cx.drop_image(image, Some(window));
                }
                if let Some(image) = self.depth.take() {
                    cx.drop_image(image, Some(window));
                }
                self.color = Some(render_image(frame.width, frame.height, &frame.color));
                self.depth = Some(render_image(
                    frame.width,
                    frame.height,
                    &capture::depth_rgb(frame),
                ));
                let center =
                    frame.depth[(frame.height / 2 * frame.width + frame.width / 2) as usize];
                self.detail = format!(
                    "{} · Center depth: {} · Native streams (not aligned)",
                    frame.serial,
                    if center == 0 {
                        "no return".into()
                    } else {
                        format!("{:.3} m", center as f32 * frame.depth_scale_m)
                    }
                );
                self.sequence = Some(sequence);
            }
        } else {
            if let Some(image) = self.color.take() {
                cx.drop_image(image, Some(window));
            }
            if let Some(image) = self.depth.take() {
                cx.drop_image(image, Some(window));
            }
            self.sequence = None;
            self.detail = "Waiting for live frames · Start scripts/start.sh in Terminal".into();
        }
    }

    fn capture(&self) {
        if self.capturing.swap(true, Ordering::Relaxed) {
            return;
        }
        let socket = self.socket.clone();
        let output = self.output.clone();
        let status = self.capture_status.clone();
        let capturing = self.capturing.clone();
        *status.lock().unwrap() = "Capturing a fresh RGB + depth pair…".into();
        std::thread::spawn(move || {
            let result = ipc::fresh_frame(&socket).and_then(|frame| capture::save(&frame, &output));
            *status.lock().unwrap() = match result {
                Ok(capture) => format!(
                    "Saved {}",
                    capture.metadata["directory"].as_str().unwrap_or("capture")
                ),
                Err(error) => format!("Capture failed: {error:#}"),
            };
            capturing.store(false, Ordering::Relaxed);
        });
    }
}

fn panel(title: &str, subtitle: &str, image: Option<Arc<RenderImage>>) -> impl IntoElement {
    div()
        .flex()
        .flex_col()
        .flex_1()
        .min_w_0()
        .gap_3()
        .p_4()
        .rounded_xl()
        .bg(rgb(0x151e2c))
        .child(
            div()
                .flex()
                .justify_between()
                .child(title.to_owned())
                .child(
                    div()
                        .text_sm()
                        .text_color(rgb(0x8c9eb7))
                        .child(subtitle.to_owned()),
                ),
        )
        .child(
            div()
                .w_full()
                .flex_1()
                .min_h_0()
                .bg(rgb(0x090f18))
                .rounded_lg()
                .overflow_hidden()
                .flex()
                .items_center()
                .justify_center()
                .when_some(image.clone(), |el, image| {
                    el.child(
                        img(image)
                            .w_full()
                            .h_full()
                            .object_fit(gpui::ObjectFit::Contain),
                    )
                })
                .when(image.is_none(), |el| {
                    el.child(div().text_color(rgb(0x72839c)).child("Waiting for camera"))
                }),
        )
}

impl Render for Viewer {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        self.update_images(window, cx);
        div().size_full().bg(rgb(0x0b1220)).text_color(rgb(0xe6edf7)).font_family("Helvetica")
            .p_6().flex().flex_col().gap_4()
            .child(div().flex().items_center().justify_between()
                .child(div().flex().flex_col().gap_1().child(div().text_2xl().child("RealSense Studio"))
                    .child(div().text_sm().text_color(rgb(0x8c9eb7)).child("D435 / COLOR + DEPTH")))
                .child(div().id("capture").px_4().py_2().rounded_lg().bg(rgb(0x267bce)).cursor_pointer()
                    .on_click(cx.listener(|this, _, _, _| this.capture()))
                    .child(if self.capturing.load(Ordering::Relaxed) { "Capturing…" } else { "Capture RGB + depth" })))
            .child(div().text_sm().text_color(rgb(if self.sequence.is_some() { 0x6fd8b0 } else { 0xffbd75 })).child(self.status.clone()))
            .child(div().flex().flex_1().min_h_0().gap_4().child(panel("Color", "RGB · 640 × 480", self.color.clone()))
                .child(panel("Depth", "Near 0.2 m → far 4 m · black = invalid", self.depth.clone())))
            .child(div().text_sm().text_color(rgb(0x8c9eb7)).child(self.detail.clone()))
            .child(div().text_sm().child(self.capture_status.lock().unwrap().clone()))
            .child(div().text_sm().text_color(rgb(0x637791)).child("MCP tools: camera_status · capture_rgbd   /   Depth saved as 16-bit PNG with meters-per-unit metadata"))
    }
}

pub fn run(socket: PathBuf, output: PathBuf) -> anyhow::Result<()> {
    Application::new().run(move |cx: &mut App| {
        let bounds = Bounds::centered(None, size(px(1220.), px(650.)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                window_min_size: Some(size(px(900.), px(640.))),
                ..Default::default()
            },
            |window, cx| {
                window.set_window_title("RealSense Studio");
                cx.new(|cx| Viewer::new(socket, output, cx))
            },
        )
        .expect("Open GPUI window");
        cx.on_window_closed(|cx| {
            if cx.windows().is_empty() {
                cx.quit();
            }
        })
        .detach();
        cx.activate(true);
    });
    Ok(())
}
