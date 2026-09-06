"""Configuration models for the InferOpt request scheduler and batching subsystem."""

from pydantic import BaseModel, ConfigDict, Field


class BatchConfig(BaseModel):
    """Configuration settings for request batching and dynamic wait window."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_batch_size: int = Field(
        default=4,
        gt=0,
        description="Maximum number of requests grouped into a single batch.",
    )
    batch_wait_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Maximum time in milliseconds to wait for additional requests to arrive.",
    )


class SchedulerConfig(BaseModel):
    """Configuration settings for asynchronous request scheduling and admission control."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_concurrency: int = Field(
        default=4,
        gt=0,
        description="Maximum number of batches executed concurrently by the backend.",
    )
    max_queue_size: int = Field(
        default=100,
        ge=0,
        description="Maximum requests allowed to wait in the priority queue before backpressure.",
    )
    max_history_size: int = Field(
        default=1000,
        ge=0,
        description="Maximum number of terminal request lifecycle records retained in memory.",
    )
    batch_config: BatchConfig = Field(
        default_factory=BatchConfig,
        description="Configuration parameters governing batch formation policy and wait window.",
    )
