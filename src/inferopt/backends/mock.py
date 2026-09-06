"""Deterministic Mock Inference Backend for testing and local development."""

import asyncio
import hashlib
import time
from typing import Final

from inferopt.backends.base import InferenceBackend
from inferopt.core.models import InferenceRequest, InferenceResponse

BACKEND_NAME: Final[str] = "mock"


def estimate_tokens(text: str) -> int:
    """Approximate token count using a deterministic whitespace splitting heuristic.

    NOTE: This is a simulation heuristic for mock testing and local development only.
    It is NOT a production tokenizer (such as tiktoken, HuggingFace tokenizers, or
    vLLM internal tokenizers). It will be replaced by actual engine tokenizer metrics
    in concrete runtime backends (e.g., MLX, vLLM).
    """
    stripped = text.strip()
    if not stripped:
        return 0
    words = stripped.split()
    return max(1, len(words))


class MockBackend(InferenceBackend):
    """Deterministic, GPU-independent mock inference engine.

    Produces predictable, reproducible outputs and token metrics for identical
    requests while supporting configurable asynchronous latency simulation.
    """

    def __init__(self, default_latency_sec: float = 0.0) -> None:
        """Initialize the mock backend.

        Args:
            default_latency_sec: Simulated execution delay in seconds (default 0.0).
        """
        if default_latency_sec < 0:
            raise ValueError("default_latency_sec cannot be negative.")
        self._default_latency_sec = default_latency_sec

    @property
    def backend_name(self) -> str:
        """Unique identifier representing this backend."""
        return BACKEND_NAME

    @property
    def default_latency_sec(self) -> float:
        """Current configured default simulated latency in seconds."""
        return self._default_latency_sec

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Execute mock inference deterministically and asynchronously.

        Args:
            request: The inference request to process.

        Returns:
            InferenceResponse containing deterministic text, token counts,
            and measured execution latency.
        """
        start_time = time.perf_counter()

        # Allow per-request latency override via metadata, falling back to backend default
        latency_sec = float(
            request.metadata.get("simulated_latency_sec", self._default_latency_sec)
        )
        if latency_sec > 0:
            await asyncio.sleep(latency_sec)

        input_tokens = estimate_tokens(request.prompt)

        # Generate deterministic output based on prompt hash and request parameters
        prompt_digest = hashlib.sha256(f"{request.model}:{request.prompt}".encode()).hexdigest()

        # Deterministically determine the number of generated tokens (up to max_tokens)
        seed_value = int(prompt_digest[:8], 16)
        desired_tokens = min(request.max_tokens, max(1, (seed_value % request.max_tokens) + 1))

        generated_words = [
            f"token_{i}_{prompt_digest[(i * 4) % 32 : (i * 4) % 32 + 4]}"
            for i in range(desired_tokens)
        ]
        generated_text = " ".join(generated_words)
        output_tokens = estimate_tokens(generated_text)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        finish_reason = "length" if output_tokens >= request.max_tokens else "stop"

        return InferenceResponse(
            request_id=request.request_id,
            generated_text=generated_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=elapsed_ms,
            backend_name=self.backend_name,
            finish_reason=finish_reason,
            metadata={
                "mock_seed": prompt_digest[:16],
                "simulated_delay_sec": latency_sec,
            },
        )
