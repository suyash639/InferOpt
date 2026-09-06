"""Configuration models for the InferOpt request scheduler."""

from pydantic import BaseModel, ConfigDict, Field


class SchedulerConfig(BaseModel):
    """Configuration settings for asynchronous request scheduling and admission control."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_concurrency: int = Field(
        default=4,
        gt=0,
        description="Maximum number of requests executed concurrently by the backend.",
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
