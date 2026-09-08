#!/usr/bin/env python3
"""World Labs API client; uses only the Python standard library."""
import argparse
import json
import os
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import HTTPRedirectHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://api.worldlabs.ai/marble/v1/"


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Keep the credential on the intended API host.


def api_key():
    key = os.environ.get("WORLDLABS_API_KEY", "").strip()
    config = ROOT / ".env.worldlabs"
    if not key and config.exists():
        for line in config.read_text().splitlines():
            if line.startswith("WORLDLABS_API_KEY="):
                key = line.partition("=")[2].strip()
                break
    if not key:
        raise ValueError("Set WORLDLABS_API_KEY or add it to .env.worldlabs.")
    return key


def request(path, payload=None):
    key = api_key()
    req = Request(
        BASE + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"WLT-Api-Key": key, "Content-Type": "application/json"},
    )
    try:
        with build_opener(NoRedirect()).open(req, timeout=45) as response:
            return json.load(response)
    except HTTPError as error:
        detail = error.read().decode(errors="replace").replace(key, "[REDACTED]")
        raise ValueError(f"World Labs HTTP {error.code}: {detail}") from None
    except URLError as error:
        raise ValueError(f"Unable to reach World Labs: {error.reason}") from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("credits", help="Check authentication and remaining API credits")
    listing = commands.add_parser("list", help="List API-generated worlds")
    listing.add_argument("--page-token")
    for name in ("get", "operation"):
        sub = commands.add_parser(name)
        sub.add_argument("id")
    generate = commands.add_parser("generate", help="Generate a world (uses API credits)")
    generate.add_argument("--request", required=True, type=Path, help="JSON request file matching the World Labs API")
    args = parser.parse_args()
    if args.command == "credits":
        result = request("credits")
    elif args.command == "list":
        payload = {"page_size": 10}
        if args.page_token:
            payload["page_token"] = args.page_token
        result = request("worlds:list", payload)
    elif args.command in ("get", "operation"):
        prefix = "worlds/" if args.command == "get" else "operations/"
        result = request(prefix + quote(args.id, safe=""))
    else:
        payload = json.loads(args.request.read_text())
        if not isinstance(payload, dict) or "world_prompt" not in payload:
            raise ValueError("Generation request must be an object containing world_prompt.")
        result = request("worlds:generate", payload)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
