#!/usr/bin/env python3
"""End-to-end stdio MCP test with an isolated synthetic camera helper."""
import base64
import json
import pathlib
import select
import struct
import subprocess
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
BINARY = ROOT / "target/debug/realsense-studio"


def receive(process, request_id):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        ready, _, _ = select.select([process.stdout], [], [], max(0, deadline - time.monotonic()))
        assert ready, "MCP response timed out"
        line = process.stdout.readline()
        assert line, "MCP server exited unexpectedly"
        message = json.loads(line)
        if message.get("id") == request_id:
            assert "error" not in message, message
            return message["result"]
    raise AssertionError("Missing MCP response")


def send(process, method, params=None, request_id=None):
    message = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        message["params"] = params
    if request_id is not None:
        message["id"] = request_id
    process.stdin.write((json.dumps(message) + "\n").encode())
    process.stdin.flush()
    if request_id is not None:
        return receive(process, request_id)


def main():
    with tempfile.TemporaryDirectory(prefix="rs-test-", dir="/tmp") as directory:
        # A helper must exit after its launcher ends, without needing sudo again.
        parent = subprocess.Popen(["sleep", "30"])
        watched = subprocess.Popen([
            str(BINARY), "--socket", str(pathlib.Path(directory) / "watched.sock"),
            "camera", "--demo", "--parent-pid", str(parent.pid)
        ], stderr=subprocess.DEVNULL)
        try:
            time.sleep(0.5)
            assert watched.poll() is None
            parent.terminate()
            parent.wait(timeout=5)
            assert watched.wait(timeout=5) == 0, "Orphan helper did not exit"
        finally:
            for process in (parent, watched):
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=5)
        socket = pathlib.Path(directory) / "camera.sock"
        captures = pathlib.Path(directory) / "captures"
        args = [str(BINARY), "--socket", str(socket), "--output", str(captures)]
        camera = subprocess.Popen(args + ["camera", "--demo"], stderr=subprocess.DEVNULL)
        mcp = None
        try:
            for _ in range(100):
                if socket.exists():
                    break
                assert camera.poll() is None, "Camera helper exited"
                time.sleep(0.05)
            mcp = subprocess.Popen(args + ["mcp"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)
            init = send(mcp, "initialize", {"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "smoke-test", "version": "1"}}, 1)
            assert "tools" in init["capabilities"]
            send(mcp, "notifications/initialized")
            listing = send(mcp, "tools/list", {}, 2)
            assert {tool["name"] for tool in listing["tools"]} == {"capture_rgbd", "camera_status"}
            status = send(mcp, "tools/call", {"name": "camera_status", "arguments": {}}, 3)
            assert "DEMO" in status["content"][0]["text"], status
            result = send(mcp, "tools/call", {"name": "capture_rgbd", "arguments": {}}, 4)
            assert not result.get("isError"), result
            metadata = json.loads(result["content"][0]["text"])
            assert metadata["demo"] is True
            assert metadata["depth_scale_m"] > 0
            assert metadata["aligned_to_color"] is False
            for content in result["content"][1:]:
                assert content["type"] == "image" and content["mimeType"] == "image/png"
                png = base64.b64decode(content["data"])
                assert png[:8] == b"\x89PNG\r\n\x1a\n"
                assert struct.unpack(">II", png[16:24]) == (640, 480)
            raw = pathlib.Path(metadata["depth"]).read_bytes()
            assert raw[24:26] == bytes([16, 0]), "Depth must be 16-bit grayscale"
            assert pathlib.Path(metadata["directory"], "metadata.json").exists()
            camera.terminate()
            camera.wait(timeout=5)
            failure = send(mcp, "tools/call", {"name": "capture_rgbd", "arguments": {}}, 5)
            assert failure.get("isError") is True, "Offline capture must fail, never return an old frame"
            print("PASS: MCP initialization, tool discovery, demo status, fresh capture, inline PNGs, raw Z16 PNG, metadata, and offline error")
        finally:
            if mcp is not None:
                mcp.terminate()
                mcp.wait(timeout=5)
            if camera.poll() is None:
                camera.terminate()
                camera.wait(timeout=5)


if __name__ == "__main__":
    main()
