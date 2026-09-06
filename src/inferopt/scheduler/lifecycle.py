"""Request lifecycle state tracking and timing metrics."""

from dataclasses import dataclass, field
from enum import StrEnum


class RequestStatus(StrEnum):
    """Lifecycle states for an inference request within the scheduler pipeline."""

    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

    @property
    def is_terminal(self) -> bool:
        """Return True if the state is terminal (completed, failed, or cancelled)."""
        return self in (RequestStatus.COMPLETED, RequestStatus.FAILED, RequestStatus.CANCELLED)


@dataclass
class RequestRecord:
    """Internal tracking record maintaining request lifecycle state and execution timing.

    All timing attributes record high-resolution timestamps from `time.perf_counter()`.
    """

    request_id: str
    status: RequestStatus = RequestStatus.QUEUED
    queued_at: float = field(default_factory=float)
    started_at: float | None = None
    completed_at: float | None = None
    error_message: str | None = None

    @property
    def queue_wait_ms(self) -> float | None:
        """Elapsed queue wait time in milliseconds (from queue entry to backend dispatch)."""
        if self.started_at is None:
            return None
        return max(0.0, (self.started_at - self.queued_at) * 1000.0)

    @property
    def execution_ms(self) -> float | None:
        """Elapsed execution time in milliseconds (from dispatch to completion/failure)."""
        if self.started_at is None or self.completed_at is None:
            return None
        return max(0.0, (self.completed_at - self.started_at) * 1000.0)

    @property
    def total_latency_ms(self) -> float | None:
        """Total turnaround time in milliseconds (from queue entry to completion)."""
        if self.completed_at is None:
            return None
        return max(0.0, (self.completed_at - self.queued_at) * 1000.0)

    def mark_running(self, started_at: float) -> None:
        """Transition request state from QUEUED to RUNNING."""
        if self.status != RequestStatus.QUEUED:
            raise ValueError(
                f"Cannot transition request {self.request_id} from {self.status} to RUNNING."
            )
        self.status = RequestStatus.RUNNING
        self.started_at = started_at

    def mark_completed(self, completed_at: float) -> None:
        """Transition request state from RUNNING to COMPLETED."""
        if self.status != RequestStatus.RUNNING:
            raise ValueError(
                f"Cannot transition request {self.request_id} from {self.status} to COMPLETED."
            )
        self.status = RequestStatus.COMPLETED
        self.completed_at = completed_at

    def mark_failed(self, completed_at: float, error_message: str) -> None:
        """Transition request state to FAILED."""
        if self.status.is_terminal:
            raise ValueError(
                f"Cannot transition terminal request {self.request_id} ({self.status}) to FAILED."
            )
        self.status = RequestStatus.FAILED
        self.completed_at = completed_at
        self.error_message = error_message

    def mark_cancelled(self, cancelled_at: float, error_message: str | None = None) -> None:
        """Transition request state to CANCELLED."""
        if self.status.is_terminal:
            raise ValueError(
                f"Cannot transition terminal request {self.request_id} "
                f"({self.status}) to CANCELLED."
            )
        self.status = RequestStatus.CANCELLED
        self.completed_at = cancelled_at
        self.error_message = error_message
