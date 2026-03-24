import sys
from importlib import import_module
from unittest.mock import MagicMock

import pytest

import stamp.__main__


def test_workbench_command_invokes_server(monkeypatch):
    """stamp workbench should delegate to stamp_workbench.server.serve."""
    captured: dict[str, object] = {}

    def fake_serve(*, host: str, port: int, root=None) -> None:
        captured["host"] = host
        captured["port"] = port
        captured["root"] = root

    # Ensure stamp_workbench is importable (real or mock)
    try:
        import stamp_workbench.server  # noqa: F401
    except ModuleNotFoundError:
        # Inject a minimal mock so the test is not skipped when the package
        # is not installed in the test environment.
        mock_module = MagicMock()
        mock_module.serve = fake_serve
        sys.modules.setdefault("stamp_workbench", MagicMock())
        sys.modules["stamp_workbench.server"] = mock_module

    monkeypatch.setattr("stamp_workbench.server.serve", fake_serve)
    monkeypatch.setattr(
        sys,
        "argv",
        ["stamp", "workbench", "--host", "0.0.0.0", "--port", "9011"],
    )

    stamp.__main__.main()

    assert captured == {"host": "0.0.0.0", "port": 9011, "root": None}


def test_workbench_missing_package_gives_helpful_error(monkeypatch):
    """stamp workbench should raise a descriptive error when stamp_workbench is not installed."""
    monkeypatch.setattr(sys, "argv", ["stamp", "workbench"])

    # Setting a module entry to None in sys.modules causes Python to raise
    # ModuleNotFoundError on any attempt to import it.
    with pytest.MonkeyPatch().context() as mp:
        mp.setitem(sys.modules, "stamp_workbench", None)  # type: ignore[arg-type]
        mp.setitem(sys.modules, "stamp_workbench.server", None)  # type: ignore[arg-type]
        with pytest.raises(SystemExit, match="1"):
            stamp.__main__.main()
