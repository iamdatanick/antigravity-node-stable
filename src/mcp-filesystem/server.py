"""MCP Filesystem Server — exposes file read/list tools via SSE transport."""

import glob
import logging
import os

from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-filesystem")

DATA_DIR = os.environ.get("DATA_DIR", "/data")
mcp = FastMCP("Antigravity-Filesystem")


@mcp.tool()
async def list_files(pattern: str = "*") -> str:
    """List files in the context directory matching a glob pattern."""
    import json

    search_path = os.path.join(DATA_DIR, pattern)
    files = glob.glob(search_path, recursive=True)
    result = []
    for f in sorted(files)[:200]:  # Limit to 200 results
        stat = os.stat(f)
        result.append(
            {
                "path": os.path.relpath(f, DATA_DIR),
                "size": stat.st_size,
                "is_dir": os.path.isdir(f),
            }
        )
    return json.dumps(result, indent=2)


@mcp.tool()
async def read_file(path: str, max_bytes: int = 100000) -> str:
    """Read a file from the context directory. Returns first max_bytes of content."""
    full_path = os.path.join(DATA_DIR, path)
    # Security: prevent path traversal
    real_path = os.path.realpath(full_path)
    real_data = os.path.realpath(DATA_DIR)
    if not real_path.startswith(real_data):
        return '{"error": "Path traversal denied"}'

    if not os.path.exists(real_path):
        return f'{{"error": "File not found: {path}"}}'

    try:
        with open(real_path, encoding="utf-8", errors="replace") as f:
            content = f.read(max_bytes)
        return content
    except Exception as e:
        return f'{{"error": "{str(e)}"}}'


@mcp.tool()
async def file_info(path: str) -> str:
    """Get metadata about a file (size, type, modification time)."""
    import json
    import time

    full_path = os.path.join(DATA_DIR, path)
    real_path = os.path.realpath(full_path)
    real_data = os.path.realpath(DATA_DIR)
    if not real_path.startswith(real_data):
        return '{"error": "Path traversal denied"}'

    if not os.path.exists(real_path):
        return f'{{"error": "File not found: {path}"}}'

    stat = os.stat(real_path)
    return json.dumps(
        {
            "path": path,
            "size": stat.st_size,
            "is_dir": os.path.isdir(real_path),
            "modified": time.ctime(stat.st_mtime),
            "extension": os.path.splitext(path)[1],
        }
    )


@mcp.tool()
async def search_files(query: str, extensions: str = ".csv,.json,.pdf,.xlsx,.md") -> str:
    """Search for files by name containing query string."""
    import json

    ext_list = [e.strip() for e in extensions.split(",")]
    matches = []
    for root, _dirs, files in os.walk(DATA_DIR):
        for f in files:
            if query.lower() in f.lower():
                ext = os.path.splitext(f)[1].lower()
                if not ext_list or ext in ext_list:
                    rel = os.path.relpath(os.path.join(root, f), DATA_DIR)
                    matches.append(rel)
        if len(matches) >= 100:
            break
    return json.dumps(matches)


if __name__ == "__main__":
    mcp.settings.host = "0.0.0.0"
    mcp.settings.port = 8000
    # Disable DNS rebinding protection for Docker network access
    mcp.settings.transport_security.enable_dns_rebinding_protection = False
    logger.info(f"Starting MCP Filesystem Server on 0.0.0.0:8000 (data_dir={DATA_DIR})")
    mcp.run(transport="sse")
