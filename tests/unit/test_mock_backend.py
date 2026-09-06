"""Unit tests for MockBackend functionality, determinism, and latency simulation."""

import asyncio

import pytest

from inferopt.backends.mock import MockBackend, estimate_tokens
from inferopt.core.models import InferenceRequest


class TestEstimateTokens:
    """Test suite for token estimation heuristic."""

    def test_estimate_empty_and_whitespace(self) -> None:
        """Ensure empty and whitespace strings return 0 tokens."""
        assert estimate_tokens("") == 0
        assert estimate_tokens("   \n\t  ") == 0

    def test_estimate_words(self) -> None:
        """Ensure token estimation splits on whitespace correctly."""
        assert estimate_tokens("hello") == 1
        assert estimate_tokens("Hello world from InferOpt") == 4
        assert estimate_tokens("  word1   word2   word3  ") == 3


class TestMockBackend:
    """Test suite for MockBackend behaviors and interface contracts."""

    def test_backend_name(self) -> None:
        """Verify the backend identifies as 'mock'."""
        backend = MockBackend()
        assert backend.backend_name == "mock"

    def test_negative_latency_rejected(self) -> None:
        """Verify negative latency configuration is rejected."""
        with pytest.raises(ValueError, match="default_latency_sec cannot be negative"):
            MockBackend(default_latency_sec=-0.1)

    @pytest.mark.asyncio
    async def test_deterministic_output(self) -> None:
        """Verify that identical requests generate identical responses."""
        backend = MockBackend(default_latency_sec=0.0)
        req = InferenceRequest(
            request_id="fixed-id-1",
            model="mock-llama-3",
            prompt="What is dynamic batching in LLM inference?",
            max_tokens=64,
            temperature=0.0,
        )

        resp1 = await backend.generate(req)
        resp2 = await backend.generate(req)

        assert resp1.request_id == "fixed-id-1"
        assert resp2.request_id == "fixed-id-1"
        assert resp1.generated_text == resp2.generated_text
        assert resp1.input_tokens == resp2.input_tokens
        assert resp1.output_tokens == resp2.output_tokens
        assert resp1.finish_reason == resp2.finish_reason
        assert resp1.backend_name == "mock"

    @pytest.mark.asyncio
    async def test_different_prompts_yield_distinct_outputs(self) -> None:
        """Verify that distinct prompts produce distinct deterministic outputs."""
        backend = MockBackend(default_latency_sec=0.0)
        req1 = InferenceRequest(model="mock-llama-3", prompt="First unique prompt")
        req2 = InferenceRequest(model="mock-llama-3", prompt="Second distinct prompt")

        resp1 = await backend.generate(req1)
        resp2 = await backend.generate(req2)

        assert resp1.generated_text != resp2.generated_text

    @pytest.mark.asyncio
    async def test_output_tokens_respect_max_tokens_bound(self) -> None:
        """Verify generated tokens do not exceed the requested max_tokens."""
        backend = MockBackend(default_latency_sec=0.0)

        for limit in [1, 5, 16, 128]:
            req = InferenceRequest(
                model="mock-llama-3",
                prompt="Explain prefix caching and KV cache reuse.",
                max_tokens=limit,
            )
            resp = await backend.generate(req)
            assert resp.output_tokens <= limit

    @pytest.mark.asyncio
    async def test_simulated_latency(self) -> None:
        """Verify simulated delay is applied and recorded in latency_ms."""
        # 10ms simulated latency
        backend = MockBackend(default_latency_sec=0.01)
        req = InferenceRequest(
            model="mock-model",
            prompt="Testing latency simulation.",
        )

        resp = await backend.generate(req)
        assert resp.latency_ms >= 9.0  # allow slight timer jitter tolerance

    @pytest.mark.asyncio
    async def test_per_request_latency_override(self) -> None:
        """Verify metadata simulated_latency_sec overrides backend default."""
        backend = MockBackend(default_latency_sec=0.0)
        req = InferenceRequest(
            model="mock-model",
            prompt="Testing metadata latency override.",
            metadata={"simulated_latency_sec": 0.01},
        )

        resp = await backend.generate(req)
        assert resp.latency_ms >= 9.0

    @pytest.mark.asyncio
    async def test_concurrent_async_execution(self) -> None:
        """Verify MockBackend handles concurrent execution with asyncio.gather."""
        backend = MockBackend(default_latency_sec=0.005)
        requests = [
            InferenceRequest(model="mock-model", prompt=f"Concurrent prompt {i}") for i in range(5)
        ]

        responses = await asyncio.gather(*(backend.generate(r) for r in requests))

        assert len(responses) == 5
        for req, resp in zip(requests, responses, strict=True):
            assert resp.request_id == req.request_id
            assert resp.backend_name == "mock"
