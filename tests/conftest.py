import pytest

from squidpy_ports.stalign import upstream


@pytest.fixture(scope="session")
def stalign():
    """The vendored upstream STalign module, or a skip if it is not checked out."""
    try:
        return upstream.load()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


@pytest.fixture
def record_property(record_property):
    """``record_property`` that takes numpy scalars, which xdist cannot send between workers."""
    import numpy as np

    return lambda name, value: record_property(name, value.item() if isinstance(value, np.generic) else value)
