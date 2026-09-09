"""Command-line interface for running reproducible InferOpt benchmarks."""

import argparse
import asyncio
import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from inferopt.backends.mock import MockBackend
from inferopt.benchmarks.generator import PRESET_SCENARIOS, generate_workload
from inferopt.benchmarks.models import WorkloadConfig
from inferopt.benchmarks.runner import BenchmarkRunner
from inferopt.scheduler.config import BatchConfig, SchedulerConfig

if TYPE_CHECKING:
    from inferopt.backends.base import InferenceBackend


class _ConcurrencyAction(argparse.Action):
    """Parse concurrency as a single int if 1 value, or list of ints if multiple."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[Any] | None,
        option_string: str | None = None,
    ) -> None:
        if isinstance(values, (list, tuple)):
            int_vals = [int(v) for v in values]
            if len(int_vals) == 1:
                setattr(namespace, self.dest, int_vals[0])
            else:
                setattr(namespace, self.dest, int_vals)
        elif isinstance(values, (str, int)):
            setattr(namespace, self.dest, int(values))


def build_parser() -> argparse.ArgumentParser:
    """Build command line argument parser for the benchmark harness."""
    parser = argparse.ArgumentParser(
        prog="inferopt-bench",
        description="InferOpt Benchmarking Harness - Run controlled inference serving benchmarks",
    )
    parser.add_argument(
        "--scenario",
        "-s",
        choices=list(PRESET_SCENARIOS.keys()),
        default="light",
        help="Pre-configured workload scenario to execute (default: light).",
    )
    parser.add_argument(
        "--num-requests",
        "-n",
        type=int,
        default=None,
        help="Override total request count in scenario.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible workload generation (default: 42).",
    )
    parser.add_argument(
        "--concurrency",
        "-c",
        nargs="+",
        action=_ConcurrencyAction,
        default=None,
        help="Scheduler max_concurrency override (or list of concurrencies for audit).",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=None,
        help="List of maximum batch sizes to evaluate in audit or matrix (e.g. 1 2 4 8).",
    )
    parser.add_argument(
        "--max-batch-size",
        "-b",
        type=int,
        default=None,
        help="Scheduler batch_config.max_batch_size override.",
    )
    parser.add_argument(
        "--batch-wait-ms",
        "-w",
        type=float,
        default=None,
        help="Scheduler batch_config.batch_wait_ms override.",
    )
    parser.add_argument(
        "--backend",
        choices=["mock", "mlx", "vllm"],
        default="mock",
        help="Inference backend implementation to evaluate (default: mock).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model identifier (defaults: mlx-community/Qwen2.5-0.5B-Instruct-4bit for MLX, "
        "Qwen/Qwen2.5-0.5B-Instruct for vLLM).",
    )
    parser.add_argument(
        "--validate-mlx",
        action="store_true",
        help="Run Step 8.5 controlled Direct MLX vs InferOpt validation experiment.",
    )
    parser.add_argument(
        "--validate-vllm",
        action="store_true",
        help="Run Step 10 controlled Direct vLLM vs InferOpt scientific benchmark experiment.",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Run complete 6-condition scientific MLX audit across concurrency and batch levels.",
    )
    parser.add_argument(
        "--batch-matrix",
        action="store_true",
        help="Run batching experiment matrix (concurrency 4,8,16 x batch 1,2,4,8).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="Warmup requests before timing (default: 2 for audit/vLLM, 1 otherwise).",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=None,
        help="Measured repetition trials (default: 3 for vLLM, 5 for audit, 3 otherwise).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="File path to save the JSON benchmark result.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable detailed benchmark output.",
    )
    return parser


async def run_benchmark_cli(args: argparse.Namespace) -> int:
    """Execute benchmark run with arguments parsed from CLI."""
    # Step 10: Controlled Scientific vLLM Benchmark Mode
    if args.validate_vllm:
        from inferopt.backends.vllm import DEFAULT_VLLM_MODEL_ID
        from inferopt.benchmarks.vllm_validation import (
            VLLMValidator,
            format_vllm_full_report,
        )

        model_id = args.model if args.model is not None else DEFAULT_VLLM_MODEL_ID
        validator = VLLMValidator(model_id=model_id)

        scenario_name = args.scenario if args.scenario != "light" else "concurrent_4"
        scenario_factory = PRESET_SCENARIOS.get(scenario_name, PRESET_SCENARIOS["concurrent_4"])
        scenario = scenario_factory(args.seed)

        if args.num_requests is not None and args.num_requests > 0:
            cfg_dict = scenario.config.model_dump()
            cfg_dict["num_requests"] = args.num_requests
            scenario = generate_workload(WorkloadConfig.model_validate(cfg_dict))

        warmup_count = args.warmup if args.warmup is not None else 2
        repetitions = args.repetitions if args.repetitions is not None else 3

        if args.concurrency is not None:
            if isinstance(args.concurrency, list):
                concurrencies = tuple(args.concurrency)
            else:
                concurrencies = (args.concurrency,)
        else:
            concurrencies = (1, 4, 8, 16)

        if args.batch_sizes is not None:
            batch_sizes = tuple(args.batch_sizes)
        elif args.max_batch_size is not None:
            batch_sizes = (args.max_batch_size,)
        else:
            batch_sizes = (1, 2, 4, 8)

        batch_wait_ms = args.batch_wait_ms if args.batch_wait_ms is not None else 50.0

        out_dir = args.output
        if out_dir is None:
            out_dir = "benchmarks/results/vllm"

        print("\n" + "=" * 80)
        print("  STARTING INFEROPT SCIENTIFIC VLLM BENCHMARK (Step 10)")
        print("=" * 80)
        print(f"  Model ID:      {model_id}")
        req_info = f"{len(scenario.requests)} requests, seed={args.seed}"
        print(f"  Scenario:      {scenario.scenario_name} ({req_info})")
        print(f"  Concurrency:   {concurrencies}")
        print(f"  Batch Sizes:   {batch_sizes}")
        print(f"  Repetitions:   {repetitions} (Warmup: {warmup_count})")
        print(f"  Output Dir:    {out_dir}")
        print("=" * 80 + "\n")

        report = await validator.run_scientific_benchmark(
            scenario=scenario,
            concurrency_levels=concurrencies,
            batch_sizes=batch_sizes,
            warmup_count=warmup_count,
            repetitions=repetitions,
            batch_wait_ms=batch_wait_ms,
            output_dir=out_dir,
        )

        print()
        print(format_vllm_full_report(report))
        print(f"\nSaved structured vLLM benchmark JSON report to: {out_dir}")
        return 0 if report.integrity.is_valid else 1

    # Step 8.5: Controlled MLX Validation / Scientific Audit Mode
    if args.validate_mlx:
        from inferopt.backends.mlx import DEFAULT_MODEL_ID as DEFAULT_MLX_MODEL_ID
        from inferopt.benchmarks.mlx_validation import (
            MLXValidator,
            format_audit_full_report,
            format_comparison_table,
            format_matrix_table,
        )

        mlx_model_id = args.model if args.model is not None else DEFAULT_MLX_MODEL_ID
        validator_mlx = MLXValidator(model_id=mlx_model_id)

        # Audit Mode: 6-Condition Scientific Matrix across Concurrency & Batch Sizes
        if args.audit:
            scenario_name = args.scenario if args.scenario != "light" else "concurrent_4"
            scenario_factory = PRESET_SCENARIOS.get(scenario_name, PRESET_SCENARIOS["concurrent_4"])
            scenario = scenario_factory(args.seed)

            # Apply request count override if specified
            if args.num_requests is not None and args.num_requests > 0:
                cfg_dict = scenario.config.model_dump()
                cfg_dict["num_requests"] = args.num_requests
                scenario = generate_workload(WorkloadConfig.model_validate(cfg_dict))

            warmup_count = args.warmup if args.warmup is not None else 2
            repetitions = args.repetitions if args.repetitions is not None else 5

            if args.concurrency is not None:
                if isinstance(args.concurrency, list):
                    concurrencies = tuple(args.concurrency)
                else:
                    concurrencies = (args.concurrency,)
            else:
                concurrencies = (1, 4, 8)

            if args.batch_sizes is not None:
                batch_sizes = tuple(args.batch_sizes)
            elif args.max_batch_size is not None:
                batch_sizes = (args.max_batch_size,)
            else:
                batch_sizes = (1, 2, 4, 8)

            out_path = args.output
            if out_path is None:
                out_path = f"benchmarks/results/mlx_audit/{scenario.scenario_name}.json"

            print("\n" + "=" * 80)
            print("  STARTING INFEROPT SCIENTIFIC MLX AUDIT (Step 8.5)")
            print("=" * 80)
            print(f"  Model ID:      {mlx_model_id}")
            req_info = f"{len(scenario.requests)} requests, seed={args.seed}"
            print(f"  Scenario:      {scenario.scenario_name} ({req_info})")
            print(f"  Concurrency:   {concurrencies}")
            print(f"  Batch Sizes:   {batch_sizes}")
            print(f"  Repetitions:   {repetitions} (Warmup: {warmup_count})")
            print(f"  Output JSON:   {out_path}")
            print("=" * 80 + "\n")

            report_audit = await validator_mlx.run_audit_experiment(
                scenario=scenario,
                concurrencies=concurrencies,
                batch_sizes=batch_sizes,
                warmup_count=warmup_count,
                repetitions=repetitions,
                output_path=out_path,
            )

            print()
            print(format_audit_full_report(report_audit))
            print(f"\nSaved structured audit JSON report to: {out_path}")
            return 0 if report_audit.integrity.is_valid else 1

        if args.batch_matrix:
            warmup_count = args.warmup if args.warmup is not None else 1
            if args.concurrency is not None:
                if isinstance(args.concurrency, list):
                    concurrencies = tuple(args.concurrency)
                else:
                    concurrencies = (args.concurrency,)
            else:
                concurrencies = (4, 8, 16)
            batch_sizes = tuple(args.batch_sizes) if args.batch_sizes is not None else (1, 2, 4, 8)

            print("\nExecuting InferOpt MLX Batching Experiment Matrix...")
            matrix_report = await validator_mlx.run_batch_matrix(
                concurrencies=concurrencies,
                batch_sizes=batch_sizes,
                warmup_count=warmup_count,
                output_path=args.output,
            )
            print()
            print(format_matrix_table(matrix_report))
            if args.output:
                print(f"\nSaved batch matrix results to: {args.output}")
            return 0

        # Legacy 2-way validation mode
        val_factory = PRESET_SCENARIOS.get(args.scenario)
        if val_factory is None:
            print(f"Error: Unknown scenario '{args.scenario}'", file=sys.stderr)
            return 1
        scenario = val_factory(args.seed)

        warmup_count = args.warmup if args.warmup is not None else 1
        repetitions = args.repetitions if args.repetitions is not None else 3

        print(f"\nExecuting MLX Validation Experiment on '{scenario.scenario_name}'...")
        print(f"Model: {mlx_model_id} | Repetitions: {repetitions} | Warmup: {warmup_count}")

        default_sched = SchedulerConfig()
        max_batch_size = (
            args.max_batch_size
            if args.max_batch_size is not None
            else default_sched.batch_config.max_batch_size
        )
        batch_wait_ms = (
            args.batch_wait_ms
            if args.batch_wait_ms is not None
            else default_sched.batch_config.batch_wait_ms
        )
        if args.concurrency is not None:
            max_concurrency = (
                args.concurrency[0] if isinstance(args.concurrency, list) else args.concurrency
            )
        else:
            max_concurrency = default_sched.max_concurrency

        sched_config = SchedulerConfig(
            max_concurrency=max_concurrency,
            batch_config=BatchConfig(max_batch_size=max_batch_size, batch_wait_ms=batch_wait_ms),
        )

        out_path = args.output
        if out_path is None:
            out_path = f"benchmarks/results/mlx_validation/{scenario.scenario_name}.json"

        report_legacy = await validator_mlx.run_validation_experiment(
            scenario=scenario,
            warmup_count=warmup_count,
            repetitions=repetitions,
            scheduler_config=sched_config,
            output_path=out_path,
        )

        print()
        print(format_comparison_table(report_legacy))
        print(f"\nSaved validation report to: {out_path}")
        return 0 if report_legacy.correctness_gate.is_valid else 1

    main_factory = PRESET_SCENARIOS.get(args.scenario)
    if main_factory is None:
        print(f"Error: Unknown scenario '{args.scenario}'", file=sys.stderr)
        return 1

    scenario = main_factory(args.seed)

    # Apply request count override if specified
    if args.num_requests is not None and args.num_requests > 0:
        cfg_dict = scenario.config.model_dump()
        cfg_dict["num_requests"] = args.num_requests
        scenario = generate_workload(WorkloadConfig.model_validate(cfg_dict))

    # Build scheduler config overrides
    default_sched = SchedulerConfig()
    default_batch = default_sched.batch_config

    max_batch_size = (
        args.max_batch_size if args.max_batch_size is not None else default_batch.max_batch_size
    )
    batch_wait_ms = (
        args.batch_wait_ms if args.batch_wait_ms is not None else default_batch.batch_wait_ms
    )
    if args.concurrency is not None:
        max_concurrency = (
            args.concurrency[0] if isinstance(args.concurrency, list) else args.concurrency
        )
    else:
        max_concurrency = default_sched.max_concurrency

    batch_cfg = BatchConfig(max_batch_size=max_batch_size, batch_wait_ms=batch_wait_ms)
    sched_config = SchedulerConfig(max_concurrency=max_concurrency, batch_config=batch_cfg)

    backend: InferenceBackend
    if args.backend == "mlx":
        from inferopt.backends.mlx import DEFAULT_MODEL_ID as DEFAULT_MLX_MODEL_ID
        from inferopt.backends.mlx import MLXBackend

        backend = MLXBackend(model_id=args.model or DEFAULT_MLX_MODEL_ID)
    elif args.backend == "vllm":
        from inferopt.backends.vllm import DEFAULT_VLLM_MODEL_ID, VLLMBackend

        backend = VLLMBackend(model=args.model or DEFAULT_VLLM_MODEL_ID)
    else:
        backend = MockBackend(default_latency_sec=0.005)

    runner = BenchmarkRunner()

    if args.verbose:
        print(f"Executing scenario: {scenario.scenario_name} ({len(scenario.requests)} reqs)...")

    result = await runner.run(
        scenario=scenario,
        backend=backend,
        scheduler_config=sched_config,
        metadata={"cli": True, "seed": args.seed, "backend": args.backend},
    )

    separator = "=" * 60
    sub_sep = "-" * 60

    print()
    print(separator)
    print(f"  InferOpt Benchmark Results: {result.scenario_name.upper()}")
    print(separator)
    req_summary = f"  Requests:       {result.completed_requests}/{result.total_requests} completed"
    if result.failed_requests > 0:
        req_summary += f" ({result.failed_requests} failed)"
    print(req_summary)
    print(f"  Duration:       {result.duration_sec:.3f} s")
    tput_str = (
        f"  Throughput:     {result.requests_per_sec:.2f} req/s | "
        f"{result.batches_per_sec:.2f} batch/s | "
        f"{result.tokens_per_sec:.1f} tokens/s"
    )
    print(tput_str)
    print(sub_sep)
    print(f"  Latency (avg):  {result.avg_latency_ms:.2f} ms")
    print(f"  Latency (p50):  {result.p50_latency_ms:.2f} ms")
    print(f"  Latency (p95):  {result.p95_latency_ms:.2f} ms")
    print(f"  Latency (p99):  {result.p99_latency_ms:.2f} ms")
    print(f"  Queue Wait:     {result.avg_queue_wait_ms:.2f} ms")
    print(f"  Execution:      {result.avg_execution_ms:.2f} ms")
    print(sub_sep)
    batch_str = (
        f"  Total Batches:  {result.total_batches} "
        f"(avg size {result.avg_batch_size:.2f}, max {result.max_batch_size})"
    )
    print(batch_str)
    print(f"  Peak Queue:     {result.peak_queue_depth} requests")
    print(f"  Peak Active:    {result.peak_active_requests} requests")
    print(separator)
    print()

    if args.output:
        result.save_json(args.output)
        if args.verbose:
            print(f"Saved benchmark result to: {args.output}")

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Main entrypoint for CLI execution."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return asyncio.run(run_benchmark_cli(args))


if __name__ == "__main__":
    sys.exit(main())
