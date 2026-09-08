mod camera;
mod capture;
mod ipc;
mod mcp;
mod viewer;

use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(about = "D435 color/depth viewer and MCP capture server")]
struct Args {
    /// Camera helper socket. Must match across helper, viewer, and MCP.
    #[arg(long, global = true, default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/.runtime/camera.sock"))]
    socket: PathBuf,
    /// Captures are written by the viewer/MCP, never by the privileged helper.
    #[arg(long, global = true, default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/captures"))]
    output: PathBuf,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Native GPUI viewer (default).
    Viewer,
    /// Camera-only helper; on macOS run with sudo for USB access.
    Camera {
        /// Synthetic test pattern; never accesses the camera.
        #[arg(long)]
        demo: bool,
        /// Exit when the launcher process ends (used by scripts/start.sh).
        #[arg(long)]
        parent_pid: Option<i32>,
    },
    /// MCP server over stdio; connects to the running camera helper.
    Mcp,
    /// Save a fresh color/depth pair without opening the viewer.
    Capture,
    /// Report camera helper status.
    Status,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    match args.command.unwrap_or(Command::Viewer) {
        Command::Camera { demo, parent_pid } => {
            if let Some(pid) = parent_pid {
                anyhow::ensure!(pid > 1, "Invalid launcher PID");
                std::thread::spawn(move || {
                    loop {
                        // Signal 0 checks existence without sending a signal.
                        if unsafe { libc::kill(pid, 0) } == -1
                            && std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH)
                        {
                            eprintln!("Launcher exited; stopping camera helper.");
                            std::process::exit(0);
                        }
                        std::thread::sleep(std::time::Duration::from_millis(250));
                    }
                });
            }
            camera::serve(&args.socket, demo)
        }
        Command::Mcp => {
            tokio::runtime::Runtime::new()?.block_on(mcp::serve(args.socket, args.output))
        }
        Command::Capture => {
            let frame = ipc::fresh_frame(&args.socket)?;
            let capture = capture::save(&frame, &args.output)?;
            println!("{}", serde_json::to_string_pretty(&capture.metadata)?);
            Ok(())
        }
        Command::Status => {
            println!(
                "{}",
                ipc::request(&args.socket, ipc::Request::Status)?.status
            );
            Ok(())
        }
        Command::Viewer => viewer::run(args.socket, args.output),
    }
}
