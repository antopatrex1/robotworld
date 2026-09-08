use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use std::{
    io::{Read, Write},
    os::unix::net::UnixStream,
    path::Path,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

const MAX_MESSAGE: u64 = 8 * 1024 * 1024;

#[derive(Clone, Serialize, Deserialize)]
pub struct Frame {
    pub sequence: u64,
    pub captured_unix_ms: u64,
    pub width: u32,
    pub height: u32,
    pub color: Vec<u8>,
    pub depth: Vec<u16>,
    pub depth_scale_m: f32,
    pub color_timestamp_ms: f64,
    pub depth_timestamp_ms: f64,
    pub color_frame_number: u64,
    pub depth_frame_number: u64,
    pub serial: String,
    pub demo: bool,
}

impl Frame {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.width > 0 && self.width <= 1920 && self.height > 0 && self.height <= 1080,
            "Invalid frame dimensions"
        );
        let count = self.width as usize * self.height as usize;
        ensure!(
            self.color.len() == count * 3 && self.depth.len() == count,
            "Invalid frame buffers"
        );
        ensure!(
            self.depth_scale_m.is_finite() && self.depth_scale_m > 0.,
            "Invalid depth scale"
        );
        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
pub enum Request {
    Status,
    Latest,
    Fresh,
}

#[derive(Serialize, Deserialize)]
pub struct Response {
    pub status: String,
    pub frame: Option<Frame>,
}

pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

pub fn write_message<T: Serialize>(stream: &mut UnixStream, value: &T) -> Result<()> {
    let bytes = bincode::serialize(value)?;
    ensure!(bytes.len() as u64 <= MAX_MESSAGE, "Message too large");
    stream.write_all(&(bytes.len() as u32).to_le_bytes())?;
    stream.write_all(&bytes)?;
    Ok(())
}

pub fn read_message<T: serde::de::DeserializeOwned>(stream: &mut UnixStream) -> Result<T> {
    use bincode::Options;
    let mut size = [0; 4];
    stream.read_exact(&mut size)?;
    let size = u32::from_le_bytes(size) as u64;
    ensure!(size <= MAX_MESSAGE, "Message too large");
    let mut bytes = vec![0; size as usize];
    stream.read_exact(&mut bytes)?;
    Ok(bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_limit(MAX_MESSAGE)
        .reject_trailing_bytes()
        .deserialize(&bytes)?)
}

pub fn request(path: &Path, request: Request) -> Result<Response> {
    let mut stream = UnixStream::connect(path).context("Camera helper unavailable. Start scripts/start.sh in Terminal (macOS USB access requires sudo).")?;
    stream.set_read_timeout(Some(Duration::from_secs(8)))?;
    stream.set_write_timeout(Some(Duration::from_secs(3)))?;
    write_message(&mut stream, &request)?;
    let response: Response = read_message(&mut stream)?;
    if let Some(frame) = &response.frame {
        frame.validate()?;
    }
    Ok(response)
}

pub fn fresh_frame(path: &Path) -> Result<Frame> {
    let response = request(path, Request::Fresh)?;
    match response.frame {
        Some(frame) => Ok(frame),
        None => bail!("{}", response.status),
    }
}
