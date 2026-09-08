"""MLX inference backend implementation for Apple Silicon acceleration."""

import argparse
import asyncio
import time
from typing import Any, Final

from inferopt.backends.base import BatchInferenceBackend, InferenceBackend
from inferopt.core.exceptions import BackendError, InferenceError
from inferopt.core.models import InferenceBatch, InferenceRequest, InferenceResponse

BACKEND_NAME: Final[str] = "mlx"
DEFAULT_MODEL_ID: Final[str] = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"


class MLXBackend(InferenceBackend, BatchInferenceBackend):
    """Hardware-accelerated inference backend executing on Apple Silicon via MLX.

    Supports lazy model loading, deterministic generation, real tokenizer token
    accounting, asynchronous thread dispatch, and genuine backend-level batched
    generation via mlx-lm.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        default_temperature: float = 0.0,
        default_max_tokens: int = 128,
        tokenizer_config: dict[str, Any] | None = None,
        model_config: dict[str, Any] | None = None,
        adapter_path: str | None = None,
    ) -> None:
        """Initialize the MLX inference backend.

        Args:
            model_id: HuggingFace repository or local path of the MLX model.
            default_temperature: Default sampling temperature (0.0 for deterministic decoding).
            default_max_tokens: Default upper bound on generated tokens.
            tokenizer_config: Optional tokenizer configuration overrides.
            model_config: Optional model configuration overrides.
            adapter_path: Optional path to LoRA adapter weights.
        """
        self._model_id = model_id
        self._default_temperature = max(0.0, default_temperature)
        self._default_max_tokens = max(1, default_max_tokens)
        self._tokenizer_config = dict(tokenizer_config or {})
        self._model_config = dict(model_config or {})
        self._adapter_path = adapter_path

        self._model: Any = None
        self._tokenizer: Any = None
        self._lock = asyncio.Lock()

    @property
    def backend_name(self) -> str:
        """Unique identifier representing this backend."""
        return BACKEND_NAME

    @property
    def model_id(self) -> str:
        """Configured model repository or directory path."""
        return self._model_id

    @property
    def default_temperature(self) -> float:
        """Default sampling temperature."""
        return self._default_temperature

    @property
    def default_max_tokens(self) -> int:
        """Default maximum tokens to generate."""
        return self._default_max_tokens

    @property
    def is_loaded(self) -> bool:
        """True if the model and tokenizer are currently loaded in memory."""
        return self._model is not None and self._tokenizer is not None

    async def load_model(self) -> None:
        """Lazily load the MLX model and tokenizer into memory once.

        Thread-safe and idempotent. Reuses loaded weights across requests.

        Raises:
            BackendError: If MLX dependencies are missing or model loading fails.
        """
        if self.is_loaded:
            return

        async with self._lock:
            if self.is_loaded:
                return

            def _load_sync() -> tuple[Any, Any]:
                try:
                    import mlx_lm
                except ImportError as err:
                    raise BackendError(
                        "MLX dependencies are not installed. "
                        "Install with: pip install 'inferopt[mlx]'"
                    ) from err

                try:
                    load_result = mlx_lm.load(
                        self._model_id,
                        tokenizer_config=self._tokenizer_config or None,
                        model_config=self._model_config or None,
                        adapter_path=self._adapter_path,
                    )
                    return load_result[0], load_result[1]
                except Exception as exc:
                    raise BackendError(
                        f"Failed to load MLX model '{self._model_id}': {exc}"
                    ) from exc

            self._model, self._tokenizer = await asyncio.to_thread(_load_sync)

    async def unload_model(self) -> None:
        """Release loaded model and tokenizer resources from memory."""
        async with self._lock:
            self._model = None
            self._tokenizer = None
            try:
                import mlx.core as mx

                if hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
                    mx.metal.clear_cache()
            except ImportError:
                pass

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Execute a single inference generation request on Apple Silicon.

        Args:
            request: Validated domain inference request.

        Returns:
            InferenceResponse containing generated text, exact token counts, and latency.

        Raises:
            InferenceError: If model execution or tokenization fails.
            BackendError: If MLX is not installed or model fails to load.
        """
        await self.load_model()

        max_tokens = request.max_tokens if request.max_tokens > 0 else self._default_max_tokens
        temperature = (
            request.temperature if request.temperature >= 0.0 else self._default_temperature
        )

        def _generate_sync() -> tuple[str, int, int, float]:
            try:
                import mlx_lm
                import mlx_lm.sample_utils
            except ImportError as err:
                raise BackendError(
                    "MLX dependencies are not installed. Install with: pip install 'inferopt[mlx]'"
                ) from err

            t0 = time.perf_counter()

            # Exact prompt token count from tokenizer
            try:
                if hasattr(self._tokenizer, "encode"):
                    prompt_tokens = len(self._tokenizer.encode(request.prompt))
                else:
                    prompt_tokens = 0
            except Exception:
                prompt_tokens = 0

            try:
                sampler = mlx_lm.sample_utils.make_sampler(temp=temperature)
                generated_text = mlx_lm.generate(
                    self._model,
                    self._tokenizer,
                    prompt=request.prompt,
                    max_tokens=max_tokens,
                    sampler=sampler,
                    verbose=False,
                )
            except Exception as exc:
                raise InferenceError(
                    f"MLX generation failed for request '{request.request_id}': {exc}"
                ) from exc

            latency_ms = max(0.0, (time.perf_counter() - t0) * 1000.0)

            # Exact output token count from tokenizer
            try:
                if hasattr(self._tokenizer, "encode"):
                    output_tokens = len(self._tokenizer.encode(generated_text))
                else:
                    output_tokens = 0
            except Exception:
                output_tokens = 0

            return generated_text, prompt_tokens, output_tokens, latency_ms

        try:
            text, in_tok, out_tok, lat_ms = await asyncio.to_thread(_generate_sync)
            return InferenceResponse(
                request_id=request.request_id,
                generated_text=text,
                input_tokens=in_tok,
                output_tokens=out_tok,
                latency_ms=lat_ms,
                backend_name=self.backend_name,
                metadata=dict(request.metadata),
            )
        except (InferenceError, BackendError):
            raise
        except Exception as exc:
            raise InferenceError(
                f"Unexpected failure executing MLX generation for '{request.request_id}': {exc}"
            ) from exc

    async def generate_batch(self, batch: InferenceBatch) -> list[InferenceResponse]:
        """Execute genuine backend-level batched inference on Apple Silicon.

        Args:
            batch: Validated domain inference batch.

        Returns:
            List of InferenceResponse instances corresponding to each request in the batch.

        Raises:
            InferenceError: If batched execution fails.
            BackendError: If MLX is not installed or model fails to load.
        """
        await self.load_model()

        def _batch_generate_sync() -> list[tuple[str, int, int, float]]:
            try:
                import mlx_lm
            except ImportError as err:
                raise BackendError(
                    "MLX dependencies are not installed. Install with: pip install 'inferopt[mlx]'"
                ) from err

            t0 = time.perf_counter()

            encoded_prompts: list[list[int]] = []
            max_tokens_list: list[int] = []

            for req in batch.requests:
                try:
                    tokens = self._tokenizer.encode(req.prompt)
                    encoded_prompts.append(tokens)
                except Exception as exc:
                    raise InferenceError(
                        f"Tokenizer failed encoding prompt for request '{req.request_id}': {exc}"
                    ) from exc

                req_max_tokens = req.max_tokens if req.max_tokens > 0 else self._default_max_tokens
                max_tokens_list.append(req_max_tokens)

            try:
                batch_response = mlx_lm.batch_generate(
                    self._model,
                    self._tokenizer,
                    prompts=encoded_prompts,
                    max_tokens=max_tokens_list,
                    verbose=False,
                )
            except Exception as exc:
                raise InferenceError(
                    f"MLX batch_generate failed for batch '{batch.batch_id}': {exc}"
                ) from exc

            batch_latency_ms = max(0.0, (time.perf_counter() - t0) * 1000.0)
            results: list[tuple[str, int, int, float]] = []

            for idx, (_req, generated_text) in enumerate(
                zip(batch.requests, batch_response.texts, strict=False)
            ):
                in_tok = len(encoded_prompts[idx])
                try:
                    out_tok = len(self._tokenizer.encode(generated_text))
                except Exception:
                    out_tok = 0
                results.append((generated_text, in_tok, out_tok, batch_latency_ms))

            return results

        try:
            batch_results = await asyncio.to_thread(_batch_generate_sync)
            return [
                InferenceResponse(
                    request_id=req.request_id,
                    generated_text=text,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    latency_ms=lat_ms,
                    backend_name=self.backend_name,
                    metadata=dict(req.metadata),
                )
                for req, (text, in_tok, out_tok, lat_ms) in zip(
                    batch.requests, batch_results, strict=False
                )
            ]
        except (InferenceError, BackendError):
            raise
        except Exception as exc:
            raise InferenceError(
                f"Unexpected failure executing MLX batch generation for '{batch.batch_id}': {exc}"
            ) from exc


def _cli_smoke_test() -> None:
    """CLI smoke-test entry point for local Apple Silicon validation."""
    parser = argparse.ArgumentParser(
        description="InferOpt MLX Backend Smoke Test (Apple Silicon Local Validation)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=f"Target MLX model identifier (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain what InferOpt does in one sentence.",
        help="Prompt text to generate completions for",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32,
        help="Maximum generation tokens (default: 32)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0 for deterministic output)",
    )
    args = parser.parse_args()

    async def _run() -> None:
        print("=" * 60)
        print("InferOpt MLX Backend Smoke Test")
        print("=" * 60)
        print(f"Model ID:    {args.model}")
        print(f"Prompt:      {args.prompt}")
        print(f"Max Tokens:  {args.max_tokens}")
        print(f"Temperature: {args.temperature}")
        print("-" * 60)
        print("Loading MLX model and tokenizer...")

        backend = MLXBackend(
            model_id=args.model,
            default_temperature=args.temperature,
            default_max_tokens=args.max_tokens,
        )
        t_load_start = time.perf_counter()
        await backend.load_model()
        t_load_ms = (time.perf_counter() - t_load_start) * 1000.0
        print(f"Model loaded successfully in {t_load_ms:.2f}ms.")

        req = InferenceRequest(
            request_id="smoke-test-1",
            model=args.model,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )

        print("-" * 60)
        print("Executing inference generation...")
        t_gen_start = time.perf_counter()
        response = await backend.generate(req)
        t_total_ms = (time.perf_counter() - t_gen_start) * 1000.0

        out_tokens = response.output_tokens or 0
        tps = (out_tokens / (response.latency_ms / 1000.0)) if response.latency_ms > 0 else 0.0

        print("-" * 60)
        print(f"Generated Output:\n{response.generated_text}")
        print("-" * 60)
        print(f"Input Tokens:      {response.input_tokens}")
        print(f"Output Tokens:     {response.output_tokens}")
        print(f"Backend Latency:   {response.latency_ms:.2f}ms")
        print(f"Total Turnaround:  {t_total_ms:.2f}ms")
        print(f"Output Throughput: {tps:.2f} tokens/sec")
        print("=" * 60)

    asyncio.run(_run())


if __name__ == "__main__":
    _cli_smoke_test()
