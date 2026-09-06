"""Smoke test to verify InferOpt package initialization and importability."""

import inferopt


def test_inferopt_version() -> None:
    """Verify that the inferopt package imports and exposes a valid version string."""
    assert hasattr(inferopt, "__version__")
    assert isinstance(inferopt.__version__, str)
    assert inferopt.__version__ == "0.1.0"


def test_submodules_importable() -> None:
    """Verify that all architectural subpackages are present and importable."""
    import inferopt.api
    import inferopt.backends
    import inferopt.config
    import inferopt.core
    import inferopt.optimizer
    import inferopt.router
    import inferopt.scheduler
    import inferopt.telemetry

    assert inferopt.api is not None
    assert inferopt.backends is not None
    assert inferopt.config is not None
    assert inferopt.core is not None
    assert inferopt.optimizer is not None
    assert inferopt.router is not None
    assert inferopt.scheduler is not None
    assert inferopt.telemetry is not None
