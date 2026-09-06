"""Base protocol defining the interface for all InferOpt inference backends."""

from typing import Protocol, runtime_checkable

from inferopt.core.models import InferenceRequest, InferenceResponse


@runtime_checkable
class InferenceBackend(Protocol):
    """Abstract protocol for model execution engines (Mock, MLX, vLLM, etc.)."""

    @property
    def backend_name(self) -> str:
        """Unique identifier representing the backend engine implementation."""
        ...

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Execute an inference generation request asynchronously.

        Args:
            request: The validated domain inference request.

        Returns:
            InferenceResponse containing generated content, token counts, and latency.

        Raises:
            InferenceError: If generation fails during backend execution.
        """
        ...
