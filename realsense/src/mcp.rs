use crate::{capture, ipc};
use base64::Engine;
use rmcp::{
    ServerHandler, ServiceExt,
    handler::server::router::tool::ToolRouter,
    model::{CallToolResult, ContentBlock, ServerCapabilities, ServerInfo},
    tool, tool_handler, tool_router,
};
use std::path::PathBuf;

#[derive(Clone)]
struct CameraMcp {
    socket: PathBuf,
    output: PathBuf,
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl CameraMcp {
    #[tool(
        description = "Check whether the D435 camera helper is streaming. Reports demo mode explicitly."
    )]
    async fn camera_status(&self) -> String {
        let socket = self.socket.clone();
        match tokio::task::spawn_blocking(move || ipc::request(&socket, ipc::Request::Status)).await
        {
            Ok(Ok(response)) => response.status,
            Ok(Err(e)) => format!("{e:#}"),
            Err(e) => e.to_string(),
        }
    }

    #[tool(
        description = "Capture a fresh D435 color and depth pair from one frameset. Saves color.png, raw 16-bit depth.png, depth-preview.png and metadata.json. Returns color and colorized depth images inline, plus file paths and the meters-per-depth-unit scale. Native depth is not aligned to color. Requires the camera helper; synthetic data is explicitly marked demo."
    )]
    async fn capture_rgbd(&self) -> CallToolResult {
        let socket = self.socket.clone();
        let output = self.output.clone();
        let result = tokio::task::spawn_blocking(move || {
            let frame = ipc::fresh_frame(&socket)?;
            capture::save(&frame, &output)
        })
        .await;
        match result {
            Ok(Ok(capture)) => {
                let base64 = base64::engine::general_purpose::STANDARD;
                CallToolResult::success(vec![
                    ContentBlock::text(capture.metadata.to_string()),
                    ContentBlock::image(base64.encode(capture.color_png), "image/png"),
                    ContentBlock::image(base64.encode(capture.depth_preview_png), "image/png"),
                ])
            }
            Ok(Err(e)) => CallToolResult::error(vec![ContentBlock::text(format!("{e:#}"))]),
            Err(e) => CallToolResult::error(vec![ContentBlock::text(e.to_string())]),
        }
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for CameraMcp {
    fn get_info(&self) -> ServerInfo {
        let mut info = ServerInfo::default();
        info.capabilities = ServerCapabilities::builder().enable_tools().build();
        info.instructions = Some("Use camera_status to check the helper, then capture_rgbd for a fresh color/depth pair. Synthetic frames are labeled demo. Raw depth is Z16 scaled to meters by metadata, and is not aligned to RGB.".into());
        info
    }
}

pub async fn serve(socket: PathBuf, output: PathBuf) -> anyhow::Result<()> {
    let server = CameraMcp {
        socket,
        output,
        tool_router: CameraMcp::tool_router(),
    };
    server
        .serve(rmcp::transport::stdio())
        .await?
        .waiting()
        .await?;
    Ok(())
}
