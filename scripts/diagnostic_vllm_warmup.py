#!/usr/bin/env python3
"""InferOpt vLLM Triton JIT Warmup & Steady-State Latency Diagnostic Tool.

Executes controlled, phase-by-phase validation of vLLM engine initialization,
single-request warmup, multi-batch warmup (B=1,2,4,8), and subsequent measured inference.
Proves whether Triton kernel_unified_attention JIT compilation occurs during Warmup
versus during Timed/Measured requests on NVIDIA Tesla T4 / Turing GPUs.
"""

import argparse
import asyncio
import os
import subprocess
import sys
import time
from typing import Any

from inferopt.backends.vllm import DEFAULT_VLLM_MODEL_ID, VLLMBackend, VLLMConfig
from inferopt.benchmarks.generator import get_concurrent_4_workload
from inferopt.benchmarks.runner import BenchmarkRunner
from inferopt.core.models import InferenceBatch, InferenceRequest
from inferopt.scheduler.config import BatchConfig, SchedulerConfig


def print_banner(text: str) -> None:
    """Print formatted section header."""
    print(f"\n{'=' * 80}\n  {text}\n{'=' * 80}")


async def run_diagnostic(
    model_id: str = DEFAULT_VLLM_MODEL_ID,
    enforce_eager: bool = True,
    gpu_mem_util: float = 0.80,
) -> int:
    """Execute step-by-step diagnostic on Tesla T4 / vLLM 0.29.0."""
    print_banner(f"INFEROPT TRITON JIT DIAGNOSTIC (PID={os.getpid()})")
    print(f"Model ID:              {model_id}")
    print(f"Enforce Eager Mode:    {enforce_eager}")
    print(f"GPU Memory Util:       {gpu_mem_util}")

    # Check GPU info via nvidia-smi
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        if smi.returncode == 0:
            print(f"NVIDIA GPU Info:       {smi.stdout.strip()}")
    except Exception:
        pass

    config = VLLMConfig(
        model=model_id,
        gpu_memory_utilization=gpu_mem_util,
        enforce_eager=enforce_eager,
        default_max_tokens=32,
        default_temperature=0.0,
    )

    backend = VLLMBackend(config=config)
    phase_events: list[dict[str, Any]] = []

    try:
        # ---------------------------------------------------------------------
        # Phase 1: Engine Initialization
        # ---------------------------------------------------------------------
        print_banner("PHASE 1: Engine Initialization (vllm.LLM construction)")
        t0 = time.perf_counter()
        await backend.load_model()
        init_ms = (time.perf_counter() - t0) * 1000.0
        print(f"[Phase 1 Complete] Engine loaded in {init_ms:.2f}ms")
        phase_events.append({"phase": "Phase 1: Engine Init", "duration_ms": init_ms})

        # ---------------------------------------------------------------------
        # Phase 2: Single-Request Warmup Probe (B=1)
        # ---------------------------------------------------------------------
        print_banner("PHASE 2: Single-Request Warmup Probe (B=1, max_tokens=32)")
        req_single = InferenceRequest(
            request_id="warmup-single-1",
            model=model_id,
            prompt=(
                "Explain the difference between thread and process execution in operating systems."
            ),
            max_tokens=32,
            temperature=0.0,
        )
        t0 = time.perf_counter()
        resp_single = await backend.generate(req_single)
        p2_ms = (time.perf_counter() - t0) * 1000.0
        print(
            f"[Phase 2 Complete] Single-request probe: input={resp_single.input_tokens} tok, "
            f"output={resp_single.output_tokens} tok, latency={p2_ms:.2f}ms"
        )
        print(f"  Response Preview: {resp_single.generated_text[:60]!r}...")
        phase_events.append({"phase": "Phase 2: Single Warmup (B=1)", "duration_ms": p2_ms})

        # ---------------------------------------------------------------------
        # Phase 3: Multi-Batch Warmup Probes (B=2, 4, 8)
        # ---------------------------------------------------------------------
        print_banner("PHASE 3: Multi-Batch Warmup Probes (B=2, B=4, B=8)")
        prompts = [
            "What is the primary function of an asynchronous event loop in Python?",
            "Describe the role of the Key-Value (KV) cache in transformer inference.",
            "List three standard cache replacement policies and describe their trade-offs.",
            "Explain how priority queueing prevents head-of-line blocking in schedulers.",
            "What is the distinction between prefill and decode during LLM generation?",
            "Define backpressure in distributed queueing architectures.",
            "Summarize the benefits of dynamic batching in three bullet points.",
            "Provide a one-sentence definition of Time-To-First-Token (TTFT).",
        ]

        for b_size in (2, 4, 8):
            batch_reqs = tuple(
                InferenceRequest(
                    request_id=f"warmup-b{b_size}-{i}",
                    model=model_id,
                    prompt=prompts[i % len(prompts)],
                    max_tokens=32,
                    temperature=0.0,
                )
                for i in range(b_size)
            )
            batch = InferenceBatch(
                batch_id=f"warmup-batch-{b_size}",
                requests=batch_reqs,
            )
            t0 = time.perf_counter()
            batch_resps = await backend.generate_batch(batch)
            b_ms = (time.perf_counter() - t0) * 1000.0
            tot_in = sum(r.input_tokens for r in batch_resps)
            tot_out = sum(r.output_tokens for r in batch_resps)
            print(
                f"[Phase 3 Probe B={b_size}] Batch completed: in={tot_in} tok, out={tot_out} tok, "
                f"latency={b_ms:.2f}ms ({b_ms / b_size:.2f}ms/req)"
            )
            phase_events.append(
                {"phase": f"Phase 3: Batch Warmup (B={b_size})", "duration_ms": b_ms}
            )

        # ---------------------------------------------------------------------
        # Phase 4: First Timed Measured Single Request (Steady-State B=1)
        # ---------------------------------------------------------------------
        print_banner("PHASE 4: First Measured Single Request (Steady-State B=1, Timed)")
        req_measured = InferenceRequest(
            request_id="measured-single-1",
            model=model_id,
            prompt="What is Amdahl's Law and how does it apply to parallel computing limits?",
            max_tokens=32,
            temperature=0.0,
        )
        t0 = time.perf_counter()
        resp_measured = await backend.generate(req_measured)
        p4_ms = (time.perf_counter() - t0) * 1000.0
        print(
            f"[Phase 4 Complete] Measured single request: input={resp_measured.input_tokens} tok, "
            f"output={resp_measured.output_tokens} tok, latency={p4_ms:.2f}ms"
        )
        print(f"  Response Preview: {resp_measured.generated_text[:60]!r}...")
        phase_events.append({"phase": "Phase 4: Measured Single (B=1)", "duration_ms": p4_ms})

        # ---------------------------------------------------------------------
        # Phase 5: First Timed Measured Batch Request (Steady-State B=4)
        # ---------------------------------------------------------------------
        print_banner("PHASE 5: First Measured Batch Request (Steady-State B=4, Timed)")
        batch_measured_reqs = tuple(
            InferenceRequest(
                request_id=f"measured-b4-{i}",
                model=model_id,
                prompt=prompts[i],
                max_tokens=32,
                temperature=0.0,
            )
            for i in range(4)
        )
        batch_measured = InferenceBatch(
            batch_id="measured-batch-4",
            requests=batch_measured_reqs,
        )
        t0 = time.perf_counter()
        batch_resps_meas = await backend.generate_batch(batch_measured)
        p5_ms = (time.perf_counter() - t0) * 1000.0
        tot_in_m = sum(r.input_tokens for r in batch_resps_meas)
        tot_out_m = sum(r.output_tokens for r in batch_resps_meas)
        print(
            f"[Phase 5 Complete] Measured batch (B=4): in={tot_in_m} tok, out={tot_out_m} tok, "
            f"latency={p5_ms:.2f}ms ({p5_ms / 4:.2f}ms/req)"
        )
        phase_events.append({"phase": "Phase 5: Measured Batch (B=4)", "duration_ms": p5_ms})

        # ---------------------------------------------------------------------
        # Phase 6: Multi-Request Concurrent InferOpt Dynamic Batching (C=4, B=4)
        # ---------------------------------------------------------------------
        print_banner("PHASE 6: Concurrent InferOpt Dynamic Batching (C=4, B=4, 8 Requests)")
        scenario = get_concurrent_4_workload(seed=42)
        sched_cfg = SchedulerConfig(
            max_concurrency=4,
            batch_config=BatchConfig(max_batch_size=4, batch_wait_ms=50.0),
        )
        runner = BenchmarkRunner()
        t0 = time.perf_counter()
        res_sched = await runner.run(scenario=scenario, backend=backend, scheduler_config=sched_cfg)
        p6_dur_s = time.perf_counter() - t0
        print(
            f"[Phase 6 Complete] Scheduler completed {res_sched.completed_requests} requests "
            f"in {p6_dur_s:.3f}s ({res_sched.requests_per_sec:.2f} req/s, "
            f"avg_latency={res_sched.avg_latency_ms:.2f}ms, "
            f"batches={res_sched.total_batches}, avg_batch_size={res_sched.avg_batch_size:.1f})"
        )
        phase_events.append(
            {"phase": "Phase 6: Scheduler Concurrent (C=4)", "duration_ms": p6_dur_s * 1000.0}
        )

    finally:
        print_banner("PHASE 7: Engine Teardown & VRAM Cleanup")
        t0 = time.perf_counter()
        await backend.unload_model()
        teardown_ms = (time.perf_counter() - t0) * 1000.0
        print(f"[Phase 7 Complete] Engine unloaded and VRAM released in {teardown_ms:.2f}ms")

    # -------------------------------------------------------------------------
    # Diagnostic Summary Table
    # -------------------------------------------------------------------------
    print_banner("DIAGNOSTIC SUMMARY & LATENCY COMPARISON")
    print(f"{'Execution Phase':<38} {'Latency (ms)':>14}")
    print("-" * 54)
    for evt in phase_events:
        print(f"{evt['phase']:<38} {evt['duration_ms']:>14.2f} ms")
    print("=" * 54)
    print("\nDIAGNOSIS INSTRUCTIONS:")
    print("1. Review the console logs from each phase.")
    print("2. If 'WARNING: Triton kernel JIT compilation during inference' was logged,")
    print("   check which Phase it appeared under:")
    print("   • If it appeared during Phase 2 or Phase 3: Warmup absorbed the one-time JIT cost.")
    print("   • If Phase 4/5 had significantly lower latency than Phase 2/3 without warnings:")
    print("     Steady-state inference was achieved!")
    print("=" * 54 + "\n")
    return 0


def main() -> None:
    """CLI entry point for diagnostic tool."""
    parser = argparse.ArgumentParser(
        description="InferOpt vLLM Triton JIT Warmup & Steady-State Diagnostic"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_VLLM_MODEL_ID,
        help=f"Model identifier (default: {DEFAULT_VLLM_MODEL_ID})",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        default=True,
        help="Enforce eager execution mode (diagnostic mode).",
    )
    parser.add_argument(
        "--no-enforce-eager",
        dest="enforce_eager",
        action="store_false",
        help="Enable CUDA graphs / normal mode.",
    )
    parser.add_argument(
        "--gpu-mem",
        type=float,
        default=0.80,
        help="GPU memory utilization fraction (default: 0.80).",
    )
    args = parser.parse_args()

    sys.exit(
        asyncio.run(
            run_diagnostic(
                model_id=args.model,
                enforce_eager=args.enforce_eager,
                gpu_mem_util=args.gpu_mem,
            )
        )
    )


if __name__ == "__main__":
    main()
