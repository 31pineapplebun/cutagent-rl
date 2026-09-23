"""Serve the static M10A human review bundle on loopback only."""

from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path("artifacts/m10/human_calibration/review_bundle"),
    )
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    root = args.bundle.resolve(strict=True)
    if not (root / "index.html").is_file():
        raise FileNotFoundError("bundle index.html is missing")
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    handler = partial(SimpleHTTPRequestHandler, directory=str(root))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"Serving human review bundle at http://127.0.0.1:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
