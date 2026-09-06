"""Unit tests for core domain models (InferenceRequest and InferenceResponse)."""

from typing import Any

import pytest
from pydantic import ValidationError

from inferopt.core.models import InferenceRequest, InferenceResponse


class TestInferenceRequest:
    """Test suite for InferenceRequest validation and behaviors."""

    def test_valid_request_minimal(self) -> None:
        """Verify minimal valid request with default parameters."""
        req = InferenceRequest(
            model="meta-llama/Llama-3-8B-Instruct",
            prompt="Hello, world!",
        )
        assert req.model == "meta-llama/Llama-3-8B-Instruct"
        assert req.prompt == "Hello, world!"
        assert isinstance(req.request_id, str)
        assert len(req.request_id) > 0
        assert req.max_tokens == 128
        assert req.temperature == 0.7
        assert req.priority == 0
        assert req.metadata == {}

    def test_valid_request_explicit_all_fields(self) -> None:
        """Verify explicit instantiation across all fields."""
        req = InferenceRequest(
            request_id="req-12345",
            model="mistralai/Mistral-7B",
            prompt="Explain KV caching.",
            max_tokens=256,
            temperature=0.0,
            priority=5,
            metadata={"user_id": "u-42", "tier": "premium"},
        )
        assert req.request_id == "req-12345"
        assert req.model == "mistralai/Mistral-7B"
        assert req.prompt == "Explain KV caching."
        assert req.max_tokens == 256
        assert req.temperature == 0.0
        assert req.priority == 5
        assert req.metadata == {"user_id": "u-42", "tier": "premium"}

    def test_empty_prompt_rejected(self) -> None:
        """Ensure empty prompt string raises ValidationError."""
        with pytest.raises(ValidationError):
            InferenceRequest(model="test-model", prompt="")

    def test_whitespace_prompt_rejected(self) -> None:
        """Ensure whitespace-only prompt raises ValidationError."""
        with pytest.raises(ValidationError, match="Prompt cannot be empty or whitespace-only"):
            InferenceRequest(model="test-model", prompt="   \n\t  ")

    def test_empty_model_rejected(self) -> None:
        """Ensure empty model string raises ValidationError."""
        with pytest.raises(ValidationError):
            InferenceRequest(model="", prompt="Valid prompt")

    def test_invalid_max_tokens_zero_or_negative(self) -> None:
        """Ensure non-positive max_tokens raises ValidationError."""
        with pytest.raises(ValidationError):
            InferenceRequest(model="test-model", prompt="Test", max_tokens=0)
        with pytest.raises(ValidationError):
            InferenceRequest(model="test-model", prompt="Test", max_tokens=-10)

    def test_invalid_temperature_bounds(self) -> None:
        """Ensure temperature outside [0.0, 2.0] raises ValidationError."""
        with pytest.raises(ValidationError):
            InferenceRequest(model="test-model", prompt="Test", temperature=-0.1)
        with pytest.raises(ValidationError):
            InferenceRequest(model="test-model", prompt="Test", temperature=2.1)

    def test_model_immutability(self) -> None:
        """Ensure InferenceRequest instances are frozen/immutable."""
        req = InferenceRequest(model="test-model", prompt="Test")
        req_any: Any = req
        with pytest.raises(ValidationError):
            req_any.prompt = "New prompt"


class TestInferenceResponse:
    """Test suite for InferenceResponse validation and behaviors."""

    def test_valid_response(self) -> None:
        """Verify valid response creation with required and default fields."""
        resp = InferenceResponse(
            request_id="req-12345",
            generated_text="KV caching stores key and value tensors.",
            input_tokens=4,
            output_tokens=7,
            latency_ms=12.5,
            backend_name="mock",
        )
        assert resp.request_id == "req-12345"
        assert resp.generated_text == "KV caching stores key and value tensors."
        assert resp.input_tokens == 4
        assert resp.output_tokens == 7
        assert resp.latency_ms == 12.5
        assert resp.backend_name == "mock"
        assert resp.finish_reason == "stop"
        assert resp.metadata == {}

    def test_invalid_negative_tokens(self) -> None:
        """Ensure negative token counts raise ValidationError."""
        with pytest.raises(ValidationError):
            InferenceResponse(
                request_id="req-1",
                generated_text="text",
                input_tokens=-1,
                output_tokens=5,
                latency_ms=10.0,
                backend_name="mock",
            )
        with pytest.raises(ValidationError):
            InferenceResponse(
                request_id="req-1",
                generated_text="text",
                input_tokens=5,
                output_tokens=-1,
                latency_ms=10.0,
                backend_name="mock",
            )

    def test_invalid_negative_latency(self) -> None:
        """Ensure negative latency raises ValidationError."""
        with pytest.raises(ValidationError):
            InferenceResponse(
                request_id="req-1",
                generated_text="text",
                input_tokens=5,
                output_tokens=5,
                latency_ms=-0.5,
                backend_name="mock",
            )

    def test_invalid_empty_backend_name(self) -> None:
        """Ensure empty backend_name raises ValidationError."""
        with pytest.raises(ValidationError):
            InferenceResponse(
                request_id="req-1",
                generated_text="text",
                input_tokens=5,
                output_tokens=5,
                latency_ms=10.0,
                backend_name="",
            )

    def test_response_serialization(self) -> None:
        """Ensure response serializes cleanly to dict and JSON."""
        resp = InferenceResponse(
            request_id="req-1",
            generated_text="output",
            input_tokens=2,
            output_tokens=1,
            latency_ms=5.0,
            backend_name="mock",
            finish_reason="length",
            metadata={"source": "test"},
        )
        data = resp.model_dump()
        assert data["request_id"] == "req-1"
        assert data["finish_reason"] == "length"
        assert data["metadata"] == {"source": "test"}
