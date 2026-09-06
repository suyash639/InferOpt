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

**Stage 0 — Project Initialization & Architecture Foundation**

InferOpt is currently in its initial architectural setup phase. Core domain boundaries, directory structures, typing configurations, and development workflows are established. Engine execution logic, schedulers, and infrastructure dependencies will be introduced incrementally in subsequent phases.

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

## Development Environment

- **Local Platform**: Developed and validated on Apple Silicon (macOS M1) with zero direct hardware coupling.
- **Backend Portability**: The system architecture enforces a backend-agnostic design using strict Python protocols. This allows full local development and testing using mock or MLX backends without requiring local NVIDIA GPU hardware, while ensuring immediate compatibility with vLLM when deployed to GPU infrastructure.
