import pytest

from squidpy_ports.stalign import upstream


@pytest.fixture(scope="session")
def stalign():
    """The vendored upstream STalign module, or a skip if it is not checked out."""
    try:
        return upstream.load()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
