from __future__ import annotations

import argparse
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .service import (
    RunManager,
    TerminalManager,
    bootstrap_payload,
    create_temp_slide_table,
    export_config_yaml,
    import_config_yaml,
    inspect_column_values,
    inspect_table,
    list_directory,
    validate_payload_dict,
)

STATIC_DIR = Path(__file__).with_name("static")
RUN_MANAGER = RunManager()
TERMINAL_MANAGER = TerminalManager()


class WorkbenchHandler(BaseHTTPRequestHandler):
    server_version = "STAMPWorkbench/0.2"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        segments = self._segments(parsed.path)

        if parsed.path == "/api/bootstrap":
            payload = bootstrap_payload()
            payload["runs"] = RUN_MANAGER.list_runs()
            payload["terminal"] = TERMINAL_MANAGER.snapshot()
            self._send_json(payload)
            return

        if parsed.path == "/api/runs":
            self._send_json({"runs": RUN_MANAGER.list_runs()})
            return

        if len(segments) == 3 and segments[:2] == ["api", "runs"]:
            run = RUN_MANAGER.get_run(segments[2])
            if run is None:
                self._send_json({"error": "Run not found."}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json(run)
            return

        if parsed.path == "/api/files":
            query = parse_qs(parsed.query)
            try:
                result = list_directory(query.get("path", [None])[0])
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(result)
            return

        if parsed.path == "/api/terminal":
            self._send_json(TERMINAL_MANAGER.snapshot())
            return

        self._serve_static(parsed.path)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        segments = self._segments(parsed.path)
        body = self._read_json_body(allow_empty=True)

        if parsed.path == "/api/validate":
            result = validate_payload_dict(body or {})
            status = HTTPStatus.OK if result["valid"] else HTTPStatus.BAD_REQUEST
            self._send_json(result, status=status)
            return

        if parsed.path == "/api/runs":
            try:
                run = RUN_MANAGER.create(body or {})
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(run, status=HTTPStatus.CREATED)
            return

        if len(segments) == 4 and segments[:2] == ["api", "runs"]:
            run_id, action = segments[2], segments[3]
            try:
                if action == "start":
                    run = RUN_MANAGER.start(run_id)
                elif action == "stop":
                    run = RUN_MANAGER.stop(run_id)
                elif action == "terminate":
                    run = RUN_MANAGER.terminate(run_id)
                else:
                    raise KeyError("Unknown run action.")
            except KeyError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
                return
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(run)
            return

        if parsed.path == "/api/analyze-table":
            try:
                result = inspect_table((body or {})["path"])
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(result)
            return

        if parsed.path == "/api/import-config":
            try:
                payload = body or {}
                result = import_config_yaml(
                    content=payload.get("content"),
                    path_str=payload.get("path"),
                    filename=payload.get("filename"),
                )
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(result)
            return

        if parsed.path == "/api/export-config":
            try:
                result = export_config_yaml(body or {})
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(result)
            return

        if parsed.path == "/api/column-values":
            try:
                payload = body or {}
                result = inspect_column_values(payload["path"], payload["column_name"])
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(result)
            return

        if parsed.path == "/api/auto-slide-table":
            try:
                result = create_temp_slide_table(body or {})
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(result)
            return

        if parsed.path == "/api/terminal":
            try:
                result = TERMINAL_MANAGER.run((body or {}).get("command", ""))
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(result)
            return

        self._send_json({"error": "Unknown endpoint."}, status=HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return

    def _read_json_body(self, *, allow_empty: bool = False) -> dict | None:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            return {} if allow_empty else None
        raw = self.rfile.read(content_length)
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return {} if allow_empty else None

    def _send_json(self, payload: dict, *, status: HTTPStatus = HTTPStatus.OK) -> None:
        content = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _serve_static(self, path: str) -> None:
        relative = "index.html" if path in {"", "/"} else path.lstrip("/")
        target = (STATIC_DIR / relative).resolve()
        if STATIC_DIR not in target.parents and target != STATIC_DIR:
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        if not target.exists() or not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        content_type, _ = mimetypes.guess_type(target.name)
        data = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header(
            "Content-Type",
            content_type or "application/octet-stream",
        )
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    @staticmethod
    def _segments(path: str) -> list[str]:
        return [segment for segment in path.split("/") if segment]


def serve(*, host: str = "127.0.0.1", port: int = 8010) -> None:
    server = ThreadingHTTPServer((host, port), WorkbenchHandler)
    print(f"STAMP workbench running at http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the STAMP workbench server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args(argv)
    serve(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
