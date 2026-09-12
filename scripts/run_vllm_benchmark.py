"""Standalone runner script for the InferOpt Scientific vLLM Benchmark (Step 10)."""

import argparse
import asyncio
import sys
from pathlib import Path

from inferopt.backends.vllm import DEFAULT_VLLM_MODEL_ID
from inferopt.benchmarks.generator import PRESET_SCENARIOS, generate_workload
from inferopt.benchmarks.models import WorkloadConfig
from inferopt.benchmarks.vllm_validation import VLLMValidator, format_vllm_full_report


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for running the scientific vLLM benchmark."""
    parser = argparse.ArgumentParser(
        description="InferOpt Scientific Real-vLLM Benchmark Harness (Step 10)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_VLLM_MODEL_ID,
        help=f"Target Hugging Face model repository (default: {DEFAULT_VLLM_MODEL_ID})",
    )
    parser.add_argument(
        "--scenario",
        choices=list(PRESET_SCENARIOS.keys()),
        default="concurrent_4",
        help="Workload scenario to replay across all conditions (default: concurrent_4)",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=None,
        help="Override total request count in workload scenario",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed for workload generation (default: 42)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=[1, 4, 8, 16],
        help="List of concurrency levels to evaluate (default: 1 4 8 16)",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="List of maximum batch sizes to evaluate (default: 1 2 4 8)",
    )
    parser.add_argument(
        "--batch-wait-ms",
        type=float,
        default=50.0,
        help="Dynamic batch formation wait window in ms (default: 50.0)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Warmup requests before timed repetitions (default: 2)",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="Measured repetition trials per condition (default: 3)",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        default=False,
        help="Enforce eager execution mode in vLLM (disables CUDA graphs, diagnostic mode)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="benchmarks/results/vllm",
        help="Directory to save JSON benchmark results (default: benchmarks/results/vllm)",
    )
    return parser


async def run_benchmark(args: argparse.Namespace) -> int:
    """Execute vLLM scientific benchmark harness."""
    scenario_factory = PRESET_SCENARIOS.get(args.scenario, PRESET_SCENARIOS["concurrent_4"])
    scenario = scenario_factory(args.seed)

    if args.num_requests is not None and args.num_requests > 0:
        cfg_dict = scenario.config.model_dump()
        cfg_dict["num_requests"] = args.num_requests
        scenario = generate_workload(WorkloadConfig.model_validate(cfg_dict))

    print("\n" + "=" * 80)
    print("  INFEROPT SCIENTIFIC REAL-VLLM BENCHMARK HARNESS (STEP 10)")
    print("=" * 80)
    print(f"  Model ID:        {args.model}")
    print(f"  Scenario:        {scenario.scenario_name} ({len(scenario.requests)} requests)")
    print(f"  Concurrency:     {args.concurrency}")
    print(f"  Batch Sizes:     {args.batch_sizes}")
    print(f"  Batch Wait:      {args.batch_wait_ms} ms")
    print(f"  Enforce Eager:   {args.enforce_eager}")
    print(f"  Repetitions:     {args.repetitions} (Warmup: {args.warmup})")
    print(f"  Output Dir:      {args.output_dir}")
    print("=" * 80 + "\n")

    validator = VLLMValidator(model_id=args.model, enforce_eager=args.enforce_eager)

    try:
        report = await validator.run_scientific_benchmark(
            scenario=scenario,
            concurrency_levels=args.concurrency,
            batch_sizes=args.batch_sizes,
            warmup_count=args.warmup,
            repetitions=args.repetitions,
            batch_wait_ms=args.batch_wait_ms,
            output_dir=Path(args.output_dir),
        )

        print()
        print(format_vllm_full_report(report))
        print(f"\nSaved structured JSON benchmark report to: {args.output_dir}")
        return 0 if report.integrity.is_valid else 1
    except Exception as exc:
        print(f"\n[ERROR] Benchmark execution failed: {exc}", file=sys.stderr)
        return 1


def main() -> None:
    """Main CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(asyncio.run(run_benchmark(args)))


if __name__ == "__main__":
    main()
