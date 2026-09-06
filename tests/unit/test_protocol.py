"""Unit tests verifying InferenceBackend protocol conformance."""

import inspect

from inferopt.backends.base import InferenceBackend
from inferopt.backends.mock import MockBackend


def test_mock_backend_implements_protocol() -> None:
    """Verify MockBackend satisfies the runtime-checkable InferenceBackend protocol."""
    backend = MockBackend()
    assert isinstance(backend, InferenceBackend)


def test_mock_backend_interface_signatures() -> None:
    """Verify required attributes and async method signatures."""
    backend = MockBackend()
    assert hasattr(backend, "backend_name")
    assert isinstance(backend.backend_name, str)
    assert hasattr(backend, "generate")
    assert inspect.iscoroutinefunction(backend.generate)
