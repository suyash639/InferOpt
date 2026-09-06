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

**Stage 6 — Deterministic Optimization Engine Foundation**

InferOpt provides an asynchronous request scheduler (`Scheduler`), dynamic batching subsystem (`BatchConfig`, `InferenceBatch`), in-process telemetry layer (`MetricsCollector`, `MetricsSnapshot`), deterministic benchmarking framework (`WorkloadScenario`, `BenchmarkRunner`), and a deterministic optimization engine (`DeterministicOptimizer`, `CandidateSpace`, `ObjectiveConfig`, `OptimizationConstraints`). The optimization engine evaluates empirical benchmark results against multi-criteria objectives and constraints, ranking configurations using a deterministic tie-breaking hierarchy and generating grid-search experiment plans.

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
    |  | & Dynamic Batching  |  |     Balancer       |  | & Benchmark (S5) |  |
    |  +----------+----------+  +---------+----------+  +--------+---------+  |
    |             |                       |                      |            |
    |             +-----------------------+----------------------+            |
    |                                     |                                   |
    |                                     v                                   |
    |                        +-------------------------+                      |
    |                        | Telemetry & Feedback    |                      |
    |                        | (MetricsCollector, ITL) |                      |
    |                        +------------+------------+                      |
    |                                     |                                   |
    |                                     v                                   |
    |                        +-------------------------+                      |
    |                        |  Deterministic Optimizer|                      |
    |                        |  & Planner Engine (S6)  |                      |
    |                        +------------+------------+                      |
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
               |      (Async Protocol)       |
               +--------------+--------------+
                              |
              +---------------+---------------+
              |                               |
              v                               v (Upcoming)
     +-----------------+             +-------------------+
     |   MockBackend   |             | MLX / vLLM Engine |
     | (Deterministic  |             | (Hardware-backed  |
     |  & GPU-free)    |             |  Execution)       |
     +-----------------+             +-------------------+
```

### Why Backend Abstraction?
Direct dependencies on specific serving engines (like vLLM or MLX) bind control-plane logic to particular hardware platforms, external C++ extensions, and heavyweight dependencies. By introducing the `InferenceBackend` protocol, InferOpt standardizes request handling, token accounting, and latency telemetry across all execution engines.

### Purpose of MockBackend
The `MockBackend` provides a deterministic, GPU-free simulation runtime that:
- Executes asynchronously without requiring local GPU acceleration or heavy model weights.
- Produces deterministic, reproducible text outputs and token metrics for any given input request.
- Simulates configurable asynchronous execution latency to test schedulers, queues, and admission control under controlled timing conditions.
- Facilitates rapid unit testing and CI pipelines on developer workstations (including Apple Silicon M1).

---

## Development Environment

- **Local Platform**: Developed and validated on Apple Silicon (macOS M1) with zero direct hardware coupling.
- **Backend Portability**: The system architecture enforces a backend-agnostic design using strict Python protocols. This allows full local development and testing using mock or MLX backends without requiring local NVIDIA GPU hardware, while ensuring immediate compatibility with vLLM when deployed to GPU infrastructure.
