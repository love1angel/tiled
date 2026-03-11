"""Entry point for the tiled language server."""

import argparse
import logging

from .server import create_server


def main():
    parser = argparse.ArgumentParser(description="tiled — TileLang Language Server")
    parser.add_argument("--tcp", action="store_true", help="Use TCP transport")
    parser.add_argument("--host", default="127.0.0.1", help="TCP host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=2087, help="TCP port (default: 2087)")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    server = create_server()

    if args.tcp:
        server.start_tcp(args.host, args.port)
    else:
        server.start_io()


if __name__ == "__main__":
    main()
