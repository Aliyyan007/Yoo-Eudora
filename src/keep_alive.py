"""Tiny HTTP server so Render's web service health check passes.

Render free-tier web services must bind to $PORT or they get killed.
This also gives external cron-job pings (cron-job.org, UptimeRobot, etc.)
an endpoint to hit to keep the instance awake.
"""
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from loguru import logger


class _Handler(BaseHTTPRequestHandler):
    def _respond(self):
        body = b"ok"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    do_GET = _respond
    do_HEAD = _respond

    def log_message(self, *args):
        pass


def start_keep_alive_server() -> None:
    """Bind 0.0.0.0:$PORT in a daemon thread. Safe no-op on failure."""
    try:
        port = int(os.getenv("PORT", "10000"))
        server = HTTPServer(("0.0.0.0", port), _Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        logger.info(f"Keep-alive HTTP server listening on 0.0.0.0:{port}")
    except Exception as e:
        logger.warning(f"Keep-alive server failed to start: {e}")
