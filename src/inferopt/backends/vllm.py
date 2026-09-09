"""vLLM inference backend integration for high-throughput GPU serving."""

import argparse
import asyncio
import time
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field

from inferopt.backends.base import BatchInferenceBackend, InferenceBackend
from inferopt.core.exceptions import BackendError, InferenceError
from inferopt.core.models import InferenceBatch, InferenceRequest, InferenceResponse

BACKEND_NAME: Final[str] = "vllm"
DEFAULT_VLLM_MODEL_ID: Final[str] = "Qwen/Qwen2.5-0.5B-Instruct"


class VLLMConfig(BaseModel):
    """Strongly-typed immutable configuration model for vLLM engine parameters."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(
        default=DEFAULT_VLLM_MODEL_ID,
        min_length=1,
        description="HuggingFace model repository identifier or local directory path.",
    )
    max_model_len: int | None = Field(
        default=None,
        gt=0,
        description="Optional maximum sequence length supported by the model.",
    )
    gpu_memory_utilization: float = Field(
        default=0.9,
        gt=0.0,
        le=1.0,
        description="Fraction of GPU VRAM allocated for model weights and KV cache.",
    )
    tensor_parallel_size: int = Field(
        default=1,
        ge=1,
        description="Number of GPUs used for distributed tensor parallelism.",
    )
    dtype: str = Field(
        default="auto",
        description="Data type for model weights and activations ('auto', 'float16', 'bfloat16').",
    )
    trust_remote_code: bool = Field(
        default=False,
        description="Whether to permit execution of custom model code from repository.",
    )
    seed: int | None = Field(
        default=42,
        description="Random seed for reproducible token sampling and KV cache initialization.",
    )
    default_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Default sampling temperature (0.0 for deterministic greedy decoding).",
    )
    default_max_tokens: int = Field(
        default=128,
        gt=0,
        description="Default maximum tokens to generate per request.",
    )
    extra_engine_args: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional keyword arguments passed to the vLLM LLM constructor.",
    )
    extra_sampling_args: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional keyword arguments passed to vLLM SamplingParams.",
    )


class VLLMBackend(InferenceBackend, BatchInferenceBackend):
    """Production-quality backend executing inference via the vLLM engine.

    Conforms to both InferenceBackend and BatchInferenceBackend protocols.
    Provides lazy thread-safe initialization, deterministic sampling defaults,
    genuine native batched inference, real prompt/output token accounting,
    and safe lifecycle management.
    """

    def __init__(
        self,
        config: VLLMConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the vLLM backend with validated configuration.

        Args:
            config: Optional pre-constructed VLLMConfig model.
            **kwargs: Configuration arguments used to construct VLLMConfig if config is None.
        """
        if config is not None:
            self._config = config
        elif kwargs:
            self._config = VLLMConfig(**kwargs)
        else:
            self._config = VLLMConfig()

        self._llm: Any = None
        self._model_load_time_ms: float = 0.0
        self._lock = asyncio.Lock()

    @property
    def backend_name(self) -> str:
        """Unique identifier representing this backend engine."""
        return BACKEND_NAME

    @property
    def config(self) -> VLLMConfig:
        """Active immutable configuration parameters."""
        return self._config

    @property
    def model_id(self) -> str:
        """Configured model repository or directory path."""
        return self._config.model

    @property
    def default_temperature(self) -> float:
        """Default sampling temperature."""
        return self._config.default_temperature

    @property
    def default_max_tokens(self) -> int:
        """Default maximum tokens to generate."""
        return self._config.default_max_tokens

    @property
    def is_loaded(self) -> bool:
        """True if the vLLM engine is currently initialized in memory."""
        return self._llm is not None

    @property
    def model_load_time_ms(self) -> float:
        """Elapsed initialization latency in milliseconds for loading the engine."""
        return self._model_load_time_ms

    async def load_model(self) -> None:
        """Lazily initialize the vLLM engine in a worker thread.

        Thread-safe, non-blocking to the asyncio event loop, and idempotent.
        Reuses the single initialized LLM engine instance across requests.

        Raises:
            BackendError: If vLLM dependencies are missing or initialization fails.
        """
        if self.is_loaded:
            return

        async with self._lock:
            if self.is_loaded:
                return

            def _load_sync() -> tuple[Any, float]:
                try:
                    import vllm  # type: ignore[import-not-found]
                except ImportError as err:
                    raise BackendError(
                        "vLLM dependencies are not installed. "
                        "Install with: pip install 'inferopt[vllm]'"
                    ) from err

                t0 = time.perf_counter()
                engine_kwargs: dict[str, Any] = {
                    "model": self._config.model,
                    "gpu_memory_utilization": self._config.gpu_memory_utilization,
                    "tensor_parallel_size": self._config.tensor_parallel_size,
                    "dtype": self._config.dtype,
                    "trust_remote_code": self._config.trust_remote_code,
                }
                if self._config.max_model_len is not None:
                    engine_kwargs["max_model_len"] = self._config.max_model_len
                if self._config.seed is not None:
                    engine_kwargs["seed"] = self._config.seed

                engine_kwargs.update(self._config.extra_engine_args)

                try:
                    llm_instance = vllm.LLM(**engine_kwargs)
                    load_ms = max(0.0, (time.perf_counter() - t0) * 1000.0)
                    return llm_instance, load_ms
                except Exception as exc:
                    raise BackendError(
                        f"Failed to initialize vLLM engine for '{self._config.model}': {exc}"
                    ) from exc

            self._llm, self._model_load_time_ms = await asyncio.to_thread(_load_sync)

    async def unload_model(self) -> None:
        """Safely release the initialized vLLM engine from memory."""
        async with self._lock:
            self._llm = None
            self._model_load_time_ms = 0.0

            def _cleanup_sync() -> None:
                try:
                    import torch  # type: ignore[import-not-found]

                    if hasattr(torch, "cuda") and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass

            await asyncio.to_thread(_cleanup_sync)

    def _build_sampling_params(
        self,
        temperature: float,
        max_tokens: int,
        extra: dict[str, Any] | None = None,
    ) -> Any:
        """Construct a vLLM SamplingParams instance for request execution."""
        try:
            import vllm
        except ImportError as err:
            raise BackendError(
                "vLLM dependencies are not installed. Install with: pip install 'inferopt[vllm]'"
            ) from err

        sampling_kwargs: dict[str, Any] = {
            "temperature": max(0.0, temperature),
            "max_tokens": max(1, max_tokens),
        }
        sampling_kwargs.update(self._config.extra_sampling_args)
        if extra:
            sampling_kwargs.update(extra)

        return vllm.SamplingParams(**sampling_kwargs)

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Execute a single inference generation request via vLLM.

        Args:
            request: Validated domain inference request.

        Returns:
            InferenceResponse containing generated text, exact token counts, and latency.

        Raises:
            InferenceError: If generation execution fails.
            BackendError: If vLLM dependencies are missing or engine initialization fails.
        """
        await self.load_model()

        temperature = (
            request.temperature if request.temperature >= 0.0 else self._config.default_temperature
        )
        max_tokens = (
            request.max_tokens if request.max_tokens > 0 else self._config.default_max_tokens
        )

        def _generate_sync() -> tuple[str, int, int, float, str]:
            t0 = time.perf_counter()
            sampling_params = self._build_sampling_params(
                temperature=temperature,
                max_tokens=max_tokens,
            )

            try:
                outputs = self._llm.generate(
                    prompts=[request.prompt],
                    sampling_params=sampling_params,
                    use_tqdm=False,
                )
            except Exception as exc:
                raise InferenceError(
                    f"vLLM generation failed for request '{request.request_id}': {exc}"
                ) from exc

            lat_ms = max(0.0, (time.perf_counter() - t0) * 1000.0)

            if not outputs:
                raise InferenceError(
                    f"vLLM returned empty output list for request '{request.request_id}'"
                )

            req_out = outputs[0]
            in_tok = len(req_out.prompt_token_ids) if req_out.prompt_token_ids is not None else 0

            first_comp = req_out.outputs[0] if req_out.outputs else None
            gen_text = first_comp.text if first_comp is not None else ""
            out_tok = (
                len(first_comp.token_ids) if first_comp and first_comp.token_ids is not None else 0
            )
            finish_reason = first_comp.finish_reason or "" if first_comp else ""

            return gen_text, in_tok, out_tok, lat_ms, finish_reason

        try:
            gen_text, in_tok, out_tok, lat_ms, finish_reason = await asyncio.to_thread(
                _generate_sync
            )
            metadata = dict(request.metadata)
            if finish_reason:
                metadata["finish_reason"] = finish_reason

            return InferenceResponse(
                request_id=request.request_id,
                generated_text=gen_text,
                input_tokens=in_tok,
                output_tokens=out_tok,
                latency_ms=lat_ms,
                backend_name=self.backend_name,
                metadata=metadata,
            )
        except (InferenceError, BackendError):
            raise
        except Exception as exc:
            raise InferenceError(
                f"Unexpected failure executing vLLM generation for '{request.request_id}': {exc}"
            ) from exc

    async def generate_batch(self, batch: InferenceBatch) -> list[InferenceResponse]:
        """Execute genuine backend-level batched inference using vLLM's batch API.

        Preserves exact input sequence ordering and 1:1 request ID mapping.

        Args:
            batch: Validated domain inference batch.

        Returns:
            List of InferenceResponse instances corresponding to each request in the batch.

        Raises:
            InferenceError: If batched execution fails.
            BackendError: If vLLM dependencies are missing or engine initialization fails.
        """
        await self.load_model()

        prompts: list[str] = [req.prompt for req in batch.requests]
        sampling_params_list: list[Any] = []

        for req in batch.requests:
            temp = req.temperature if req.temperature >= 0.0 else self._config.default_temperature
            m_tok = req.max_tokens if req.max_tokens > 0 else self._config.default_max_tokens
            sampling_params_list.append(
                self._build_sampling_params(temperature=temp, max_tokens=m_tok)
            )

        def _batch_generate_sync() -> list[tuple[str, int, int, float, str]]:
            t0 = time.perf_counter()

            try:
                outputs = self._llm.generate(
                    prompts=prompts,
                    sampling_params=sampling_params_list,
                    use_tqdm=False,
                )
            except Exception as exc:
                raise InferenceError(
                    f"vLLM batch generation failed for batch '{batch.batch_id}': {exc}"
                ) from exc

            batch_latency_ms = max(0.0, (time.perf_counter() - t0) * 1000.0)
            results: list[tuple[str, int, int, float, str]] = []

            for req_out in outputs:
                in_tok = (
                    len(req_out.prompt_token_ids) if req_out.prompt_token_ids is not None else 0
                )
                first_comp = req_out.outputs[0] if req_out.outputs else None
                gen_text = first_comp.text if first_comp is not None else ""
                out_tok = (
                    len(first_comp.token_ids)
                    if first_comp and first_comp.token_ids is not None
                    else 0
                )
                finish_reason = first_comp.finish_reason or "" if first_comp else ""
                results.append((gen_text, in_tok, out_tok, batch_latency_ms, finish_reason))

            return results

        try:
            batch_results = await asyncio.to_thread(_batch_generate_sync)

            if len(batch_results) != len(batch.requests):
                raise InferenceError(
                    f"vLLM batch returned {len(batch_results)} outputs, "
                    f"expected {len(batch.requests)} for batch '{batch.batch_id}'"
                )

            responses: list[InferenceResponse] = []
            for req, (text, in_tok, out_tok, lat_ms, finish_reason) in zip(
                batch.requests, batch_results, strict=True
            ):
                metadata = dict(req.metadata)
                if finish_reason:
                    metadata["finish_reason"] = finish_reason

                responses.append(
                    InferenceResponse(
                        request_id=req.request_id,
                        generated_text=text,
                        input_tokens=in_tok,
                        output_tokens=out_tok,
                        latency_ms=lat_ms,
                        backend_name=self.backend_name,
                        metadata=metadata,
                    )
                )

            return responses
        except (InferenceError, BackendError):
            raise
        except Exception as exc:
            raise InferenceError(
                f"Unexpected failure executing vLLM batch generation for '{batch.batch_id}': {exc}"
            ) from exc


def _cli_smoke_test() -> None:
    """CLI smoke-test entry point for vLLM backend validation on NVIDIA GPUs."""
    parser = argparse.ArgumentParser(
        description="InferOpt vLLM Backend Smoke Test (NVIDIA GPU Validation)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_VLLM_MODEL_ID,
        help=f"Target vLLM model identifier (default: {DEFAULT_VLLM_MODEL_ID})",
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
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization fraction (default: 0.9)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size (default: 1)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="Model weight data type (default: auto)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from HuggingFace",
    )
    parser.add_argument(
        "--scheduler",
        action="store_true",
        help="Run end-to-end Scheduler dynamic batching smoke test (4 requests)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Max batch size for scheduler dynamic batching (default: 4)",
    )
    parser.add_argument(
        "--batch-wait-ms",
        type=float,
        default=50.0,
        help="Dynamic batch formation wait window in ms (default: 50.0)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="Max concurrency for scheduler (default: 2)",
    )
    args = parser.parse_args()

    async def _run() -> None:
        print("=" * 60)
        print("InferOpt vLLM Backend Smoke Test")
        print("=" * 60)
        print(f"Model ID:                {args.model}")
        print(f"Max Tokens:              {args.max_tokens}")
        print(f"Temperature:             {args.temperature}")
        print(f"GPU Memory Utilization:  {args.gpu_memory_utilization}")
        print(f"Tensor Parallel Size:    {args.tensor_parallel_size}")
        print(f"Data Type:               {args.dtype}")
        print(f"Scheduler Mode:          {args.scheduler}")
        print("-" * 60)
        print("Initializing vLLM engine...")

        config = VLLMConfig(
            model=args.model,
            max_model_len=None,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            default_temperature=args.temperature,
            default_max_tokens=args.max_tokens,
        )

        backend = VLLMBackend(config=config)
        t_load_start = time.perf_counter()
        await backend.load_model()
        t_load_ms = (time.perf_counter() - t_load_start) * 1000.0
        print(f"vLLM engine loaded successfully in {t_load_ms:.2f}ms.")

        if args.scheduler:
            # End-to-end scheduler dynamic batching smoke test
            from inferopt.scheduler.config import BatchConfig, SchedulerConfig
            from inferopt.scheduler.scheduler import Scheduler
            from inferopt.telemetry.collector import MetricsCollector

            scheduler_config = SchedulerConfig(
                max_concurrency=args.concurrency,
                batch_config=BatchConfig(
                    max_batch_size=args.batch_size,
                    batch_wait_ms=args.batch_wait_ms,
                ),
            )
            collector = MetricsCollector()

            prompts = [
                "Explain what InferOpt does in one sentence.",
                "What are the benefits of dynamic batching in LLM serving?",
                "How does speculative decoding improve inference throughput?",
                "Describe the role of key-value cache in transformer generation.",
            ]
            requests = [
                InferenceRequest(
                    request_id=f"vllm-e2e-smoke-{i + 1}",
                    model=args.model,
                    prompt=prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
                for i, prompt in enumerate(prompts)
            ]

            print("-" * 60)
            print(f"Submitting {len(requests)} requests through Scheduler + Dynamic Batching...")
            t_start = time.perf_counter()

            async with Scheduler(
                backend=backend,
                config=scheduler_config,
                collector=collector,
            ) as scheduler:
                responses = await asyncio.gather(*[scheduler.submit(r) for r in requests])

            t_total_sec = max(0.0001, time.perf_counter() - t_start)
            snapshot = collector.snapshot()
            req_stats = snapshot.requests
            batch_stats = snapshot.batches

            print("\nInferOpt VLLM E2E Smoke Test")
            print("----------------------------")
            print(f"Backend: {backend.__class__.__name__}")
            print(f"Requests: {req_stats.total_requests}")
            print(f"Completed: {req_stats.completed_requests}")
            print(f"Failed: {req_stats.failed_requests}")
            print(f"Batches: {batch_stats.total_batches}")
            print(f"Average batch size: {batch_stats.avg_batch_size:.1f}")
            print(f"p50 latency: {req_stats.p50_total_latency_ms:.2f} ms")
            print(f"p95 latency: {req_stats.p95_total_latency_ms:.2f} ms")
            throughput = req_stats.completed_requests / t_total_sec
            print(f"Throughput: {throughput:.2f} req/s")
            print("=" * 60)

            for resp in responses:
                print(
                    f"[{resp.request_id}] (tokens: in={resp.input_tokens}, "
                    f"out={resp.output_tokens}, lat={resp.latency_ms:.1f}ms):"
                )
                print(f"  {resp.generated_text.strip()[:120]}...\n")
        else:
            req = InferenceRequest(
                request_id="vllm-smoke-1",
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
