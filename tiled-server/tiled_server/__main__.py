"""Entry point for the tiled language server (LSP) and MCP server."""

import argparse
import logging
import sys

from . import __version__
from .server import create_server


def main():
    parser = argparse.ArgumentParser(
        description="tiled — TileLang Language Server & MCP Server",
    )
    parser.add_argument("-v", "--version", action="version", version=f"tiled {__version__}")

    sub = parser.add_subparsers(dest="command")

    # ── LSP subcommand (also the default) ──
    lsp = sub.add_parser("lsp", help="Start the Language Server (default)")
    lsp.add_argument("--tcp", action="store_true", help="Use TCP transport")
    lsp.add_argument("--stdio", action="store_true", help="Use stdio transport (default)")
    lsp.add_argument("--host", default="127.0.0.1", help="TCP host (default: 127.0.0.1)")
    lsp.add_argument("--port", type=int, default=2087, help="TCP port (default: 2087)")
    lsp.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])

    # ── MCP subcommand ──
    mcp_parser = sub.add_parser("mcp", help="Start the MCP server (stdio)")
    mcp_parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])

    # For backwards compat, allow bare flags (no subcommand = LSP)
    parser.add_argument("--tcp", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--stdio", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--host", default="127.0.0.1", help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, default=2087, help=argparse.SUPPRESS)
    parser.add_argument("--log-level", default="info", help=argparse.SUPPRESS)

    args = parser.parse_args()

    if args.command == "mcp":
        _run_mcp(args)
    else:
        _run_lsp(args)


def _run_lsp(args):
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    server = create_server()
    if args.tcp:
        server.start_tcp(args.host, args.port)
    else:
        server.start_io()


def _run_mcp(args):
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    try:
        from .mcp import mcp
    except ImportError:
        print(
            "MCP dependencies not installed. Install with:\n"
            "  pip install 'tile-lsp[mcp]'",
            file=sys.stderr,
        )
        sys.exit(1)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
