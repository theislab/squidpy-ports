import importlib

import pytest


def _vendored(method: str):
    @pytest.fixture(scope="session", name=method)
    def fixture():
        """The vendored upstream, loaded and pin-checked, or a skip if its submodule is not checked out."""
        try:
            return importlib.import_module(f"squidpy_ports.{method}.upstream").load()
        except FileNotFoundError as exc:
            pytest.skip(str(exc))

    return fixture


stalign, spage = map(_vendored, ("stalign", "spage"))
