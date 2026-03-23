import sys

import stamp.__main__


def test_workbench_command_invokes_server(monkeypatch):
    captured: dict[str, object] = {}

    def fake_serve(*, host: str, port: int) -> None:
        captured["host"] = host
        captured["port"] = port

    monkeypatch.setattr("stamp.workbench.server.serve", fake_serve)
    monkeypatch.setattr(
        sys,
        "argv",
        ["stamp", "workbench", "--host", "0.0.0.0", "--port", "9011"],
    )

    stamp.__main__.main()

    assert captured == {"host": "0.0.0.0", "port": 9011}
