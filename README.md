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

**Stage 2 — Request Scheduler Foundation & Concurrency Control**

InferOpt provides an asynchronous request scheduler (`Scheduler`) with bounded concurrency, priority-aware dispatch, deterministic FIFO tie-breaking, queue backpressure (`QueueFullError`), and request lifecycle tracking across the `InferenceBackend` protocol.

## Scheduler Architecture

The InferOpt Scheduler serves as the primary control-plane orchestrator responsible for queueing, admission control, and resource allocation across backend engines:

```text
Requests
   |
   v
Scheduler
   |
   +---- Priority Queue (FIFO Tie-Breaking)
   |
   +---- Concurrency Control (Bounded Worker Pool)
   |
   v
InferenceBackend
```

### Scheduler Responsibilities
1. **Request Ingestion & Admission**: Accepts domain `InferenceRequest` instances asynchronously via `submit()`.
2. **Backpressure & Queue Bounding**: Rejects excess traffic with `QueueFullError` when queue depth reaches `max_queue_size`.
3. **Priority Dispatch**: Prioritizes urgent workloads using `InferenceRequest.priority`.
4. **Bounded Concurrency**: Limits concurrent in-flight backend executions to `max_concurrency` via a managed worker pool.
5. **Lifecycle State Tracking**: Tracks request progress through discrete states (`QUEUED`, `RUNNING`, `COMPLETED`, `FAILED`, `CANCELLED`).
6. **Execution Telemetry**: High-resolution measurement of queue wait time and execution latency (`time.perf_counter()`).
7. **Clean Lifecycle Management**: Graceful shutdown (`start()`, `shutdown()`, and async context manager `async with Scheduler(...)`).

### Priority Model & FIFO Ordering
- **Priority Policy**: Requests with higher numeric `priority` values are dequeued and dispatched before lower-priority requests.
- **FIFO Tie-Breaking**: When multiple requests share identical priority levels, deterministic arrival-order FIFO (First-In, First-Out) is strictly preserved using monotonic sequence counters.
- **Starvation Considerations**: In pure priority scheduling, continuous high-priority traffic can starve lower-priority requests. In future optimization stages, an adaptive aging mechanism will be introduced; for this foundational layer, the strict priority model ensures deterministic SLA prioritization without heuristic overhead.

### Concurrency Control & Backpressure
- **Worker Pool**: The scheduler provisions exactly `max_concurrency` async worker tasks. Active backend executions never exceed this limit.
- **Queue Backpressure**: When `queued_requests >= max_queue_size` (or when `max_queue_size = 0` and all workers are busy), new submissions are rejected immediately with `QueueFullError` rather than buffering unboundedly or dropping requests silently.

### Request Lifecycle States
Every accepted request transitions through a strict lifecycle state machine:
- `QUEUED`: Request is waiting in the priority queue for an available worker slot.
- `RUNNING`: Request has been popped from the queue and is executing on the backend engine.
- `COMPLETED`: Backend generation finished successfully and response has been returned.
- `FAILED`: Execution failed due to a backend error (worker remains healthy; exception is isolated and returned to caller).
- `CANCELLED`: Request was cancelled by caller or during graceful shutdown before/during execution.

### Why Batching is Intentionally Deferred
Dynamic and continuous token-level batching require complex coordination with engine-level KV cache allocations, chunked prefill policies, and hardware-specific memory management. Stage 2 intentionally isolates request-level admission, priority, and concurrency control to establish a rock-solid, verifiable control-plane foundation before introducing multi-request batching in subsequent milestones.


## Planned Architecture

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
    |  | & Dynamic Batching  |  |     Balancer       |  |  & Optimizer     |  |
    |  +----------+----------+  +---------+----------+  +--------+---------+  |
    |             |                       |                      |            |
    |             +-----------------------+----------------------+            |
    |                                     |                                   |
    |                                     v                                   |
    |                        +-------------------------+                      |
    |                        | Telemetry & Feedback    |                      |
    |                        | (Metrics, KV Cache, ITL)|                      |
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

## Inference Backend Abstraction

InferOpt is designed so that the core serving and optimization components remain completely decoupled from specific model execution runtimes.

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
- Simulates configurable asynchronous execution latency (`asyncio.sleep`) to test schedulers, queues, and admission control under controlled timing conditions.
- Facilitates rapid unit testing and CI pipelines on developer workstations (including Apple Silicon M1).

Concrete backend integrations (such as MLX for Apple Silicon and vLLM for NVIDIA GPU clusters) will implement this abstraction in subsequent milestones.

## Development Environment

- **Local Platform**: Developed and validated on Apple Silicon (macOS M1) with zero direct hardware coupling.
- **Backend Portability**: The system architecture enforces a backend-agnostic design using strict Python protocols. This allows full local development and testing using mock or MLX backends without requiring local NVIDIA GPU hardware, while ensuring immediate compatibility with vLLM when deployed to GPU infrastructure.
