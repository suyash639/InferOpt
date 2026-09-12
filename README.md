# InferOpt

InferOpt is a production-oriented LLM Inference Optimization and Serving Control System designed to sit above raw inference engines to make intelligent, system-level decisions across request scheduling, dynamic batching, concurrency control, model routing, and telemetry-driven workload optimization.

## Problem

Serving large language model (LLM) inference workloads efficiently in production is fraught with structural challenges:

- **Non-deterministic generation times**: Request execution latency varies dramatically based on input prompt lengths and output generation limits.
- **KV cache memory contention**: Key-Value (KV) cache memory is finite and highly prone to fragmentation and thrashing under fluctuating concurrency.
- **Throughput vs. Latency Trade-offs**: Maximizing token throughput often leads to degradation in Time-To-First-Token (TTFT) and Inter-Token Latency (ITL).
- **Static routing and naive scheduling**: Standard load-balancing mechanisms lack awareness of engine-level cache state, prompt prefix overlap, queue depth, or hardware utilization.

Raw inference engines provide low-level execution primitives but lack global orchestration and adaptive control mechanisms across heterogeneous fleets.

## Vision

InferOpt acts as an adaptive, intelligent control and optimization layer that operates above execution engines (such as vLLM, MLX, or specialized inference runtimes). By decoupling high-level serving policy from low-level execution, InferOpt provides:

1. **Intelligent Request Scheduling & Dynamic Batching**: Policy-driven scheduling algorithms optimizing for target SLAs, token-budget constraints, and prefix caching efficiency.
2. **Backend-Agnostic Model Routing**: Real-time traffic distribution across multiple instances and engines based on queue health, cache locality, and latency metrics.
3. **Telemetry-Driven Workload Optimization**: Continuous feedback loops collecting fine-grained inference signals to adaptively modulate admission control and concurrency thresholds.
4. **Clean Decoupled Architecture**: Extensible interfaces allowing uniform control over local development engines (Mock/MLX) and high-throughput production clusters (vLLM on NVIDIA GPUs).

## Current Status

**Stage 8.5 — Real MLX Validation & Controlled Baseline**

InferOpt provides an asynchronous request scheduler (`Scheduler`), dynamic batching subsystem (`BatchConfig`, `InferenceBatch`), in-process telemetry layer (`MetricsCollector`, `MetricsSnapshot`), deterministic benchmarking framework (`WorkloadScenario`, `BenchmarkRunner`), deterministic optimization engine (`DeterministicOptimizer`, `CandidateSpace`), closed-loop adaptive controller (`AdaptiveController`, `AdaptationPolicy`), real local Apple Silicon LLM execution via `MLXBackend` (`mlx` and `mlx-lm`), and a reproducible validation framework (`MLXValidator`, `DirectMLXRunner`) comparing Direct MLX execution against InferOpt-mediated execution under identical workloads and machine conditions.

---

## Architecture

```text
                       +-----------------------------------+
                       |        Client Applications        |
                       |    (OpenAI / Custom HTTP / gRPC)  |
                       +-----------------+-----------------+
                                         |
                                         v
                       +-----------------------------------+
                       |         InferOpt API Layer        |
                       |      (FastAPI Request Ingestion)  |
                       +-----------------+-----------------+
                                         |
                                         v
    +-------------------------------------------------------------------------+
    |                         InferOpt Control Plane                          |
    |                                                                         |
    |  +---------------------+  +--------------------+  +------------------+  |
    |  |  Request Scheduler  |  |   Router & Load    |  | Workload Engine  |  |
    |  | & Dynamic Batching  |<---+ Balancer         |  | & Benchmark (S5) |  |
    |  +----------+----------+  | +---------+----------+  +--------+---------+  |
    |             |             |           |                      |            |
    |             +-------------|-----------+----------------------+            |
    |                           |           |                                   |
    |                           |           v                                   |
    |                           |  +-------------------------+                  |
    |                           |  | Telemetry & Feedback    |                  |
    |                           |  | (MetricsCollector, ITL) |                  |
    |                           |  +------------+------------+                  |
    |                           |               |                               |
    |                           |               v                               |
    |                           |  +-------------------------+                  |
    |                           |  |  Deterministic Optimizer|                  |
    |                           |  |  & Planner Engine (S6)  |                  |
    |                           |  +------------+------------+                  |
    |                           |               |                               |
    |                           |               v                               |
    |                           |  +-------------------------+                  |
    |                           +--|   Adaptive Controller   |                  |
    |         (Runtime Config      |  & Closed-Loop (S7)     |                  |
    |          Modulation)         +-------------------------+                  |
    +-------------------------------------|-----------------------------------+
                                          |
                                          v
    +-------------------------------------------------------------------------+
    |                     Backend Abstraction Protocol                        |
    |                       (InferenceBackendProtocol)                        |
    +-------------------+-----------------+--------------------+--------------+
                        |                 |                    |
                        v                 v                    v
              +-----------------+ +-----------------+ +------------------+
              |  Mock Backend   | |   MLX Backend   | |   vLLM Backend   |
              |  (Unit/CI Test) | | (Apple Silicon) | | (NVIDIA Cluster) |
              +-----------------+ +-----------------+ +------------------+
```

---

## Scheduler & Dynamic Batching

The InferOpt Scheduler serves as the primary control-plane orchestrator responsible for queueing, admission control, and resource allocation across backend engines:

```text
Requests
   |
   v
Scheduler
   |
   +---- Priority Queue (FIFO Tie-Breaking)
   |
   +---- Batch Formation (Dynamic Wait Window)
   |
   +---- Concurrency Control (Bounded Batch Workers)
   |
   +---- Telemetry Collection (Fault-Isolated Lifecycle Recording)
   |
   v
InferenceBackend / BatchInferenceBackend
```

### Scheduler Responsibilities
1. **Request Ingestion & Admission**: Accepts domain `InferenceRequest` instances asynchronously via `submit()`.
2. **Backpressure & Queue Bounding**: Rejects excess traffic with `QueueFullError` when queue depth reaches `max_queue_size`.
3. **Priority Dispatch**: Prioritizes urgent workloads using `InferenceRequest.priority`.
4. **Dynamic Batching**: Groups queued requests into `InferenceBatch` objects according to `BatchConfig`.
5. **Bounded Concurrency**: Limits concurrent in-flight batch executions to `max_concurrency`.
6. **Lifecycle State Tracking**: Tracks request progress through discrete states (`QUEUED`, `RUNNING`, `COMPLETED`, `FAILED`, `CANCELLED`).
7. **Execution Telemetry**: High-resolution measurement of queue wait time, batch formation wait, and execution latency (`time.perf_counter()`).
8. **Clean Lifecycle Management**: Graceful shutdown (`start()`, `shutdown()`, and async context manager `async with Scheduler(...)`).

### Request Lifecycle States
- `QUEUED`: Request admitted to scheduler priority queue, awaiting worker batch formation.
- `RUNNING`: Request packaged into an `InferenceBatch` and actively executing on the inference backend.
- `COMPLETED`: Backend generation finished successfully and response has been returned.
- `FAILED`: Execution failed due to a backend error (worker remains healthy; exception is isolated and returned to caller).
- `CANCELLED`: Request was cancelled by caller or during graceful shutdown before/during execution.

### Dynamic Batching Policy
- **Maximum Batch Size (`max_batch_size`)**: The upper bound of requests grouped into a single `InferenceBatch`.
- **Batching Wait Window (`batch_wait_ms`)**: When a worker begins forming a batch, if fewer than `max_batch_size` requests are immediately available, it waits up to `batch_wait_ms` for arriving traffic.
- **Immediate Dispatch**: If the queue contains `max_batch_size` requests (or reaches capacity during the window), the batch dispatches immediately with zero additional delay.
- **Partial Batch Dispatch**: If the wait window expires before reaching capacity, the partial batch is dispatched immediately.
- **Order Preservation**: Batches strictly respect the scheduler's priority and FIFO tie-breaking policies.

---

## Telemetry & Metrics Foundation

> [!IMPORTANT]
> **Step 4 telemetry is measurement infrastructure, not optimization logic.**
>
> The telemetry subsystem exists to measure, observe, and summarize serving behavior accurately and safely. It does not alter scheduling decisions, manipulate priorities, or adjust concurrency dynamically.

### What InferOpt Measures

1. **Request-Level Metrics (`RequestMetrics`)**:
   - `request_id`, `priority`, `status` (`COMPLETED`, `FAILED`, `CANCELLED`).
   - `queue_wait_ms`: Elapsed time from queue admission until backend dispatch.
   - `execution_ms`: Elapsed time from backend dispatch until response completion.
   - `total_latency_ms`: Total turnaround latency from queue submission to response resolution.
   - `input_tokens`, `output_tokens`, `max_tokens`: Token accounting.
   - `backend_name`, `batch_id`, `error_message`: Execution context.

2. **Batch-Level Metrics (`BatchMetrics`)**:
   - `batch_id`, `size`: Number of requests bundled.
   - `batch_formation_wait_ms`: Duration spent gathering requests in the formation window before dispatch.
   - `execution_ms`: Backend duration executing the entire batch.
   - `total_max_tokens`, `completed_request_count`, `failed_request_count`.

3. **Aggregate Metrics & Snapshots (`MetricsSnapshot`)**:
   - `RequestStats`: Total/completed/failed/cancelled counts, averages, and percentiles (`p50`, `p95`, `p99`).
   - `BatchStats`: Total/completed/failed batch counts, average/min/max batch sizes, formation wait, and execution time.
   - `ThroughputStats`: `requests_per_sec`, `batches_per_sec`, `tokens_per_sec`, total input/output tokens.
   - `QueueStats`: Real-time and peak queue depth and active backend concurrency.

---

## Benchmark Harness & Workload Methodology

```text
Workload Scenario (Config + Seed)
                |
                v
        BenchmarkRunner
                |
     +----------+----------+
     | Arrival Pattern:   |
     | - Sequential       |
     | - Concurrent       |
     | - Burst (Barrier)  |
     | - Fixed-Rate       |
     +----------+----------+
                |
                v
         Scheduler & Batching
                |
                v
        Inference Backend
                |
                v
       MetricsCollector
                |
                v
        BenchmarkResult (JSON)
```

### Purpose & Experimental Methodology

> [!IMPORTANT]
> **Step 5 establishes the experimental harness; it does not prove that InferOpt improves inference performance.**
>
> All workloads in Step 5 are controlled, synthetic evaluation scenarios designed to test scheduler coordination, queueing dynamics, and dynamic batching behavior under deterministic traffic profiles. Current measurements run against `MockBackend` validate control-plane logic without requiring a physical GPU. Real GPU acceleration claims require physical hardware execution and will be evaluated in subsequent milestones.

### Controlled Workload Profiles

InferOpt provides pre-configured, reproducible scenarios stored in `benchmarks/workloads/`:

1. **`light`**: 10 requests, short factual queries, `max_tokens=32`, concurrent arrival.
2. **`medium`**: 50 requests, mixed short/medium queries, `max_tokens=64`, fixed-rate arrival (10 RPS).
3. **`heavy`**: 200 requests, mixed short/medium/long prompts, `max_tokens=128`, concurrent arrival.
4. **`burst`**: 50 requests released simultaneously via an `asyncio.Event` synchronization barrier to evaluate queue backpressure and batch drain.
5. **`mixed`**: 60 requests with heterogeneous priority levels (-1, 0, 1, 2) and multiple prompt categories.
6. **`long_context`**: 20 requests with long context documents (500+ tokens) and `max_tokens=256`.

### Traffic Arrival Patterns

- **Sequential**: Submits one request at a time, awaiting resolution before issuing the next.
- **Concurrent**: Releases all requests into the scheduler queue simultaneously.
- **Burst (Barrier)**: Arms all client submission tasks behind an `asyncio.Event` barrier and releases them in a synchronized burst.
- **Fixed-Rate**: Schedules request dispatches at targeted rate offsets (`requests / rate_rps`) using monotonic timers.

### Reproducibility & Baseline Comparison

- **Deterministic Seeds**: The workload generator (`generate_workload`) relies on an explicit seed (`seed=42`). Re-running a scenario with the same seed guarantees identical prompt contents, sequence lengths, priorities, and relative arrival offsets.
- **Future Baseline Evaluation**: The same workload JSON definition can be replayed against baseline serving models versus InferOpt configurations under identical conditions to measure comparative performance.
- **Distribution Metrics**: Benchmark evaluations emphasize percentile distributions ($p50$, $p95$, $p99$) and throughput rates rather than misleading single-average figures.

### CLI Benchmark Runner

InferOpt provides a built-in CLI to execute benchmarks locally:

```bash
# Run the light benchmark scenario
python -m inferopt.benchmarks --scenario light

# Run burst benchmark with custom request count and output file
python -m inferopt.benchmarks --scenario burst --num-requests 100 -o results/burst_run.json

# Run mixed priority scenario with custom batch size and wait window
python -m inferopt.benchmarks --scenario mixed --max-batch-size 8 --batch-wait-ms 25.0
```

---

## Deterministic Optimization Engine

> [!IMPORTANT]
> **Step 6 provides deterministic optimization and recommendation infrastructure; it does not yet perform live adaptive scheduling.**
>
> The optimizer consumes empirical benchmark results (`BenchmarkResult` / `MetricsSnapshot`) or generates grid-search experiment plans across a candidate search space (`CandidateSpace`). It operates strictly as an offline evaluator and planner, ranking candidate configurations deterministically without mutating active schedulers or claiming simulated performance speedups.

```text
    +-----------------------------------------------------------+
    |                      Optimizer Input                      |
    |                                                           |
    |  +--------------------+         +----------------------+  |
    |  | Benchmark Evidence |         | Candidate Space      |  |
    |  | (BenchmarkResult)  |         | (CandidateSpace)     |  |
    |  +---------+----------+         +----------+-----------+  |
    |            |                               |              |
    +------------|-------------------------------|--------------+
                 |                               |
                 v                               v
    +-----------------------------------------------------------+
    |                 DeterministicOptimizer                    |
    |                                                           |
    |  Mode A: Historical Evaluation                            |
    |    1. Filter candidates against OptimizationConstraints   |
    |    2. Compute objective score (THROUGHPUT/LATENCY/BALANCED)|
    |    3. Deterministic tie-breaking hierarchy                |
    |    4. Emit immutable OptimizationResult                   |
    |                                                           |
    |  Mode B: Experiment Planning                              |
    |    1. Compute Cartesian product over CandidateSpace       |
    |    2. Generate structured sequence of SchedulerConfig     |
    +-----------------------------------------------------------+
```

### Optimization Objectives

The optimizer supports three transparent, deterministic scoring models (`ObjectiveConfig`):

1. **`THROUGHPUT`**: Maximizes processed request throughput ($\text{Score} = \text{requests\_per\_sec}$).
2. **`LATENCY`**: Minimizes tail latency ($\text{Score} = -\text{p95\_latency\_ms}$).
3. **`BALANCED`**: Multi-criteria weighted normalization:
   $$\text{Score} = w_{\text{tput}} \cdot \left(\frac{\text{requests\_per\_sec}}{\text{target\_throughput\_rps}}\right) - w_{\text{lat}} \cdot \left(\frac{\text{p95\_latency\_ms}}{\text{target\_p95\_latency\_ms}}\right)$$
   where $w_{\text{tput}} + w_{\text{lat}} = 1.0$ and targets normalize units across differing scales.

### Constraint Evaluation

Candidates are evaluated against explicit SLA and operational thresholds (`OptimizationConstraints`):
- `max_p95_latency_ms`: Rejects configurations whose measured $p95$ latency exceeds the SLA bound.
- `min_throughput_rps`: Rejects configurations that fail to achieve minimum required throughput.
- `max_batch_size`: Hardware memory safety bound capping maximum grouped batch size.
- `max_concurrency`: Limits concurrent in-flight worker count.
- `max_batch_wait_ms`: Prevents excessive formation delay on interactive workloads.
- `max_failed_requests`: Zero-tolerance threshold for runtime errors (default: `0`).

Candidates violating any constraint are flagged as infeasible with detailed diagnostic reasons. Infeasible candidates are ranked strictly below all feasible configurations.

### Deterministic Tie-Breaking Hierarchy

When multiple candidates achieve identical objective scores, the engine breaks ties deterministically using a fixed five-tier lexicographical comparator:

1. **Objective Score** (higher is better)
2. **$p95$ Latency** (lower is better)
3. **`batch_wait_ms`** (smaller wait window is better)
4. **`max_batch_size`** (smaller batch footprint is better)
5. **`max_concurrency`** (lower resource footprint is better)

### Example Usage

```python
from inferopt.optimizer import (
    CandidateSpace,
    DeterministicOptimizer,
    ObjectiveConfig,
    OptimizationConstraints,
    OptimizationObjectiveType,
)

optimizer = DeterministicOptimizer()

# Mode A: Evaluate historical benchmark results
objective = ObjectiveConfig(
    objective_type=OptimizationObjectiveType.BALANCED,
    throughput_weight=0.6,
    latency_weight=0.4,
    target_throughput_rps=50.0,
    target_p95_latency_ms=30.0,
)
constraints = OptimizationConstraints(max_p95_latency_ms=40.0, min_throughput_rps=20.0)

opt_result = optimizer.evaluate_results(
    benchmark_results=results,
    objective=objective,
    constraints=constraints,
)

if opt_result.is_feasible:
    print(f"Recommended Config: {opt_result.recommended_config}")
    print(f"Best Score: {opt_result.best_score:.4f}")

# Mode B: Generate grid-search experiment plan
space = CandidateSpace(
    concurrencies=(1, 2, 4),
    batch_sizes=(1, 2, 4, 8),
    batch_waits_ms=(0.0, 5.0, 10.0),
)
experiment_plan = optimizer.create_experiment_plan(space)
print(f"Generated {len(experiment_plan)} experimental configurations to run.")
```

---

## Adaptive Scheduling & Closed-Loop Control

> [!IMPORTANT]
> **Step 7 introduces deterministic closed-loop adaptation. It does not use machine learning or an LLM to make scheduling decisions.**
>
> The adaptive controller safely modulates live scheduler concurrency and dynamic batching parameters based on observed telemetry windows. All adaptation is conservative, bounded, and explainable, governed by hard anti-oscillation constraints and automated rollback safeguards.

```text
       +-------------------------------------------------------------+
       |                      Observed Telemetry                     |
       |                (MetricsSnapshot / BenchmarkResult)          |
       +------------------------------+------------------------------+
                                      |
                                      v
       +-------------------------------------------------------------+
       |                      AdaptiveController                     |
       |                                                             |
       |  1. Evaluation Window Gate (min requests & batches)         |
       |  2. Rollback Check (SLA violation / performance drop)       |
       |  3. Cooldown Counter Gate (anti-thrashing)                  |
       |  4. Candidate Ranking (via DeterministicOptimizer)          |
       |  5. Improvement Threshold & Hysteresis Gate                 |
       |  6. Safety Bounds Enforcement (min/max limits)              |
       |  7. Emits Explainable AdaptationDecision                    |
       +------------------------------+------------------------------+
                                      | (If Decision == APPLY/ROLLBACK)
                                      v
       +-------------------------------------------------------------+
       |                Scheduler.apply_config()                     |
       |                                                             |
       |  - Atomic config update                                     |
       |  - Dynamic worker pool adjustment                           |
       |  - Active tasks & queued requests fully preserved           |
       |  - Telemetry adaptation event recorded                      |
       +-------------------------------------------------------------+
```

### Closed-Loop Adaptation Architecture

1. **Evaluation Windows**: The controller never adapts on a single noisy request. Decisions require evidence thresholds (`min_completed_requests` and `min_completed_batches`). If data is insufficient, it emits an `INSUFFICIENT_DATA` decision.
2. **Improvement Threshold & Hysteresis**: Candidates must exceed a configurable relative (`min_improvement_pct`) or absolute (`min_improvement_abs`) threshold over baseline performance to prevent switching on minor metric fluctuations.
3. **Cooldown Gating**: Following any configuration update, the controller enforces a cooldown window (`cooldown_windows`) during which further switches are suppressed with a `COOLDOWN` decision.
4. **Safety Bounds Enforcement**: Candidate configurations must strictly satisfy operational bounds (`min_concurrency`, `max_concurrency`, `min_batch_size`, `max_batch_size`, `min_batch_wait_ms`, `max_batch_wait_ms`).
5. **Hard SLA & Constraint Protection**: Candidates that violate hard latency limits ($p95$) or throughput constraints are rejected with `INFEASIBLE`.
6. **Automated Rollback**: If an applied configuration subsequently violates an SLA constraint or degrades objective performance by $\ge \text{rollback\_degradation\_pct}$, the controller immediately issues a `ROLLBACK` decision and reverts the scheduler to the previous known-good configuration.

### Runtime Configuration Semantics (`apply_config`)

When `scheduler.apply_config(new_config)` is invoked on a live running `Scheduler`:
- **Atomicity**: The configuration object (`self._config`) and batch settings are atomically replaced.
- **Worker Scaling**:
  - **Scale Up**: If `max_concurrency` increases, new worker tasks are spawned immediately.
  - **Scale Down**: If `max_concurrency` decreases, excess workers finish their current batch and cleanly retire without dropping queued requests.
- **Queue Preservation**: All pending requests in the priority queue remain valid, ordered, and untouched. They are formed into subsequent batches according to the updated `BatchConfig`.
- **In-Flight Tasks**: Currently executing batches continue on backend runtimes without interruption.
- **Telemetry Observability**: An immutable `AdaptationEvent` is recorded in `MetricsCollector`.

### Example Usage

```python
from inferopt.optimizer import (
    AdaptationPolicy,
    AdaptiveController,
    ObjectiveConfig,
    OptimizationConstraints,
    OptimizationObjectiveType,
)
from inferopt.scheduler import Scheduler

# 1. Define adaptive policy
policy = AdaptationPolicy(
    objective=ObjectiveConfig(objective_type=OptimizationObjectiveType.BALANCED),
    constraints=OptimizationConstraints(max_p95_latency_ms=35.0),
    min_improvement_pct=10.0,
    cooldown_windows=2,
    min_completed_requests=30,
    min_completed_batches=5,
    enable_rollback=True,
)

# 2. Attach to running scheduler
controller = AdaptiveController(policy=policy, scheduler=scheduler)

# 3. Observe metrics snapshot and step controller
snapshot = scheduler.collector.snapshot()
decision = controller.step(snapshot, candidate_evidence=candidates)

if decision.is_applied:
    print(f"Applied new configuration: {decision.proposed_config}")
    print(f"Reason: {decision.reason}")
```

---

## Inference Backend Abstraction

```text
               +-----------------------------+
               |        InferOpt Core        |
               |  (Domain Models & Policy)   |
               +--------------+--------------+
                              |
                              v
               +-----------------------------+
               |    <<InferenceBackend>>     |
               |  <<BatchInferenceBackend>>  |
               |      (Async Protocols)      |
               +--------------+--------------+
                              |
              +---------------+---------------+
              |                               |
              v                               v
     +-----------------+             +-------------------+
     |   MockBackend   |             |    MLXBackend     |
     | (Deterministic  |             | (Apple Silicon    |
     |  & GPU-free)    |             |  Metal GPU / MLX) |
     +-----------------+             +-------------------+
```

### Why Backend Abstraction?
Direct dependencies on specific serving engines (like vLLM or MLX) bind control-plane logic to particular hardware platforms, external C++ extensions, and heavyweight dependencies. By introducing the `InferenceBackend` and `BatchInferenceBackend` protocols, InferOpt standardizes request handling, token accounting, and latency telemetry across all execution engines.

### Purpose of MockBackend
The `MockBackend` provides a deterministic, GPU-free simulation runtime that:
- Executes asynchronously without requiring local GPU acceleration or heavy model weights.
- Produces deterministic, reproducible text outputs and token metrics for any given input request.
- Simulates configurable asynchronous execution latency to test schedulers, queues, and admission control under controlled timing conditions.
- Facilitates rapid unit testing and CI pipelines on developer workstations.

---

## MLX Backend on Apple Silicon

> [!IMPORTANT]
> **Step 8 integrates real local execution on Apple Silicon; it does not claim GPU performance superiority for InferOpt.**
>
> The MLX backend executes real transformer models using Apple's `mlx` and `mlx-lm` libraries, leveraging Metal unified memory on macOS. It enables local end-to-end inference verification, tokenizer accounting, and hardware execution under the InferOpt control plane. Performance comparisons against external GPU serving engines (e.g., vLLM on NVIDIA GPUs) require subsequent cluster benchmarks.

### Key Capabilities

1. **Lazy Loading & Resource Management**: Model weights and tokenizers are loaded on demand during the first inference request with thread-safe locking (`asyncio.Lock`). Resources can be cleanly unloaded via `unload_model()`.
2. **Deterministic Sampling**: Default sampling temperature is set to `0.0` (greedy argmax) for reproducible evaluation.
3. **Exact Tokenizer Accounting**: Calculates exact prompt and generated token counts via Hugging Face tokenizers rather than synthetic approximations.
4. **Hardware-Level Batched Generation**: Single requests use `mlx_lm.generate` while formed batches execute natively on Metal using `mlx_lm.batch_generate` without synthetic loops.
5. **Zero Required Dependencies**: The core `inferopt` package has zero MLX dependencies. MLX is configured as an optional extra (`pip install "inferopt[mlx]"`).

### Installation

```bash
# Install InferOpt with Apple Silicon MLX support
pip install -e ".[mlx]"
```

### CLI Smoke Test

Verify local MLX execution with the built-in smoke test entry point:

```bash
# Run local smoke test with default model (mlx-community/Qwen2.5-0.5B-Instruct-4bit)
python -m inferopt.backends.mlx --prompt "Explain what InferOpt does in one sentence."

# Run smoke test with custom model and token limit
python -m inferopt.backends.mlx \
    --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
    --prompt "List three key features of an LLM scheduler." \
    --max-tokens 48 \
    --temperature 0.0
```

### Benchmark with MLX Backend

Run InferOpt benchmark workloads against real Apple Silicon inference:

```bash
# Run the light benchmark scenario using MLX backend
python -m inferopt.benchmarks --scenario light --backend mlx

# Run burst benchmark scenario with MLX backend and custom model
python -m inferopt.benchmarks \
    --scenario burst \
    --backend mlx \
    --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
    --num-requests 20 \
    --max-batch-size 4
```

---

## Step 8.5 — Real MLX Validation & Controlled Baseline

> [!IMPORTANT]
> **Step 8.5 validates real MLX execution and establishes a reproducible local baseline. It does not establish that InferOpt outperforms direct MLX.**
>
> The purpose of Step 8.5 is to rigorously compare Direct MLX execution (unmediated baseline) against InferOpt + MLXBackend under identical workload definitions, models, seeds, decoding parameters, warmup procedures, and machine conditions.

### Validation Methodology

```text
                  +--------------------------------+
                  | Workload Scenario (Fixed Seed) |
                  +---------------+----------------+
                                  |
            +---------------------+---------------------+
            |                                           |
            v                                           v
+-----------------------+                   +-----------------------+
|    DirectMLXRunner    |                   |    BenchmarkRunner    |
| (Unmediated Baseline) |                   | (Scheduler + Batching)|
+-----------+-----------+                   +-----------+-----------+
            |                                           |
            v                                           v
+-----------------------+                   +-----------------------+
|  DirectMLXResult (3x) |                   |  BenchmarkResult (3x) |
+-----------+-----------+                   +-----------+-----------+
            |                                           |
            +---------------------+---------------------+
                                  |
                                  v
                  +--------------------------------+
                  |         Correctness Gate       |
                  |  - Exact 1:1 ID preservation   |
                  |  - Valid non-empty output text |
                  |  - Non-negative token counts   |
                  +---------------+----------------+
                                  |
                                  v
                  +--------------------------------+
                  | Neutral Metric Comparison Table|
                  |     (p50, p95, TPS, Deltas)    |
                  +--------------------------------+
```

1. **Experimental Invariance**: Both pipelines evaluate the exact same sequence of requests, prompt texts, `max_tokens`, `temperature=0.0`, and random seed.
2. **Warmup Separation**: Executes a configurable number of warmup requests (default: 1) before timed measurement. Warmup outputs and timings are discarded and recorded in metadata.
3. **Repeated Trials**: Executes multiple measured repetitions (default: 3) for warm measurements, reporting mean, median, pooled $p50/p95/p99$ percentiles, throughput, and sample standard deviation.
4. **Correctness Gating**: Verifies completion counts, duplicate ID absence, non-empty outputs, and non-negative token counts before presenting comparisons.
5. **Batching Matrix**: Evaluates concurrency ($4, 8, 16$) across dynamic batch sizes ($1, 2, 4, 8$) to quantify hardware batching amortization on Metal unified memory.

### Running Validation Experiments

```bash
# 1. Run controlled comparison on single request scenario
python -m inferopt.benchmarks --validate-mlx --scenario single --warmup 1 --repetitions 3

# 2. Run controlled comparison on 4 concurrent requests
python -m inferopt.benchmarks --validate-mlx --scenario concurrent_4 --warmup 1 --repetitions 3

# 3. Run full batching experiment matrix (4,8,16 concurrency x 1,2,4,8 batch size)
python -m inferopt.benchmarks --validate-mlx --batch-matrix
```

All validation results are persisted in JSON format under `benchmarks/results/mlx_validation/`.

---

## vLLM Backend Integration (NVIDIA GPU Serving)

InferOpt integrates with [vLLM](https://github.com/vllm-project/vllm) for high-throughput GPU serving in production environments.

### Architectural Separation
* **vLLM as the Engine**: vLLM provides the underlying inference engine, handling low-level CUDA execution, PagedAttention, KV-cache memory management, and token generation kernels.
* **InferOpt as the Control Plane**: InferOpt sits above vLLM, managing multi-tenant prioritization, dynamic batch formation windows, bounded concurrency, backpressure, closed-loop telemetry adaptation, and model routing.
* **Native Batched Generation**: `VLLMBackend.generate_batch()` executes formed batches using vLLM's native batch generation API (`LLM.generate(prompts=..., sampling_params=...)`) without synthetic gathering loops.

### Optional Installation
vLLM is an optional dependency and is not required for core InferOpt development on Apple Silicon or CPU CI environments:

```bash
# Install InferOpt with vLLM support (requires Linux + NVIDIA CUDA GPU)
pip install -e ".[vllm]"
```

### Scientific Real-vLLM Benchmark Harness (Step 10)

InferOpt provides a scientifically controlled benchmark harness to evaluate serving overhead, batching efficiency, and latency-throughput tradeoffs against real vLLM on NVIDIA GPUs (e.g. Kaggle T4).

#### Experimental Conditions
* **Condition A (Direct vLLM)**: Unmediated execution directly invoking `vllm.LLM` without the InferOpt scheduler (baseline).
* **Condition B (InferOpt Batch 1)**: Full InferOpt scheduler and telemetry pipeline with `max_batch_size = 1`.
* **Condition C (InferOpt Batch 2)**: Dynamic batching with `max_batch_size = 2`.
* **Condition D (InferOpt Batch 4)**: Dynamic batching with `max_batch_size = 4`.
* **Condition E (InferOpt Batch 8)**: Dynamic batching with `max_batch_size = 8`.

#### Scientific Controls
* **Workload Replay & Hash**: Workloads are generated once and assigned a deterministic SHA-256 hash. The identical sequence of prompts, request IDs, and sampling parameters is replayed across every condition.
* **Warmup & Cold-Start Isolation**: Engine initialization, CUDA memory allocation, and warmup iterations are explicitly separated from steady-state timing.
* **Repetition Aggregation**: Multiple trials (default: 3) collect mean, median (p50), min, max, and sample standard deviations without cherry-picking.
* **Integrity Gate**: Automatically verifies 100% request completion, 1:1 ID preservation, non-empty outputs, real tokenizer token counts, and identical workload hashes before results are accepted.
* **Differential Overhead**: Formally calculates the end-to-end differential overhead relative to Direct vLLM.

```bash
# Run full 5-condition scientific benchmark on NVIDIA GPU (Default CUDA graph mode)
python -m inferopt.benchmarks.cli --validate-vllm \
    --concurrency 1 4 8 16 \
    --batch-sizes 1 2 4 8 \
    --repetitions 3 \
    --warmup 2

# Or run via standalone script
python scripts/run_vllm_benchmark.py --concurrency 1 4 8 16 --batch-sizes 1 2 4 8

# Diagnostic Mode: Enforce eager execution (disables CUDA graphs)
# Used for environments encountering vLLM V1 EngineCore socket initialization issues
python -m inferopt.benchmarks.cli --validate-vllm --enforce-eager
```

> [!IMPORTANT]
> **Diagnostic Mode (`--enforce-eager`) & Benchmark Comparability**:
>
> 1. **Diagnostic & Compatibility Purpose**: `--enforce-eager` is provided as an opt-in diagnostic and compatibility option for environments (such as containerized or virtualized GPU platforms like Kaggle) where default vLLM V1 engine initialization encounters IPC/socket startup failures (specifically `ValueError: b'\x00\x00' is not a valid EngineCoreRequestType` in `process_input_sockets()`).
> 2. **Not a Root Cause Fix**: Enabling eager execution bypasses CUDA graph capture and compilation, providing a working diagnostic inference path. It does *not* fix the underlying vLLM V1 EngineCore socket/IPC issue.
> 3. **Comparability Warning**: Eager execution disables `torch.compile` and CUDA graph execution, incurring per-request Python/CUDA kernel launch overhead. Therefore, results collected with `--enforce-eager` **MUST NOT** be presented as directly comparable to standard production or default vLLM performance runs.
> 4. **Production Default**: The default configuration remains `enforce_eager=False` to preserve full CUDA graph capture, compilation, and standard production serving semantics.

Results are persisted as structured JSON in `benchmarks/results/vllm/`.

---

## Development Environment

- **Local Platform**: Developed and validated on Apple Silicon (macOS) with zero direct hardware coupling in the core library.
- **Backend Portability**: The system architecture enforces a backend-agnostic design using strict Python protocols (`InferenceBackend`, `BatchInferenceBackend`). This allows full local development and testing using mock or MLX backends without requiring local NVIDIA GPU hardware, while ensuring immediate compatibility with vLLM when deployed to GPU infrastructure.



