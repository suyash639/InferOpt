"""Unit tests for MetricsCollector aggregation, percentiles, throughput, and snapshots."""

import threading

from inferopt.scheduler.lifecycle import RequestStatus
from inferopt.telemetry.collector import MetricsCollector, _compute_percentile
from inferopt.telemetry.models import BatchMetrics, RequestMetrics


class TestPercentileCalculation:
    """Tests for percentile calculation helper."""

    def test_empty_list(self) -> None:
        assert _compute_percentile([], 50.0) == 0.0

    def test_single_item(self) -> None:
        assert _compute_percentile([42.0], 50.0) == 42.0
        assert _compute_percentile([42.0], 99.0) == 42.0

    def test_known_distribution(self) -> None:
        values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        p50 = _compute_percentile(values, 50.0)
        p0 = _compute_percentile(values, 0.0)
        p100 = _compute_percentile(values, 100.0)
        assert p0 == 10.0
        assert p50 == 55.0
        assert p100 == 100.0


class TestMetricsCollector:
    """Tests for in-memory MetricsCollector recording and aggregation."""

    def test_empty_collector_snapshot(self) -> None:
        collector = MetricsCollector()
        snapshot = collector.snapshot()
        assert snapshot.requests.total_requests == 0
        assert snapshot.requests.completed_requests == 0
        assert snapshot.requests.avg_queue_wait_ms == 0.0
        assert snapshot.requests.avg_execution_ms == 0.0
        assert snapshot.requests.avg_total_latency_ms == 0.0
        assert snapshot.batches.total_batches == 0
        assert snapshot.batches.avg_batch_size == 0.0
        assert snapshot.throughput.requests_per_sec == 0.0
        assert snapshot.queue.peak_queue_depth == 0

    def test_record_single_request(self) -> None:
        collector = MetricsCollector()
        collector.record_request(
            RequestMetrics(
                request_id="req-1",
                priority=1,
                queue_wait_ms=10.0,
                execution_ms=20.0,
                total_latency_ms=30.0,
                status=RequestStatus.COMPLETED,
                input_tokens=15,
                output_tokens=25,
                max_tokens=64,
                backend_name="mock-backend",
            )
        )
        snapshot = collector.snapshot()
        assert snapshot.requests.total_requests == 1
        assert snapshot.requests.completed_requests == 1
        assert snapshot.requests.failed_requests == 0
        assert snapshot.requests.cancelled_requests == 0
        assert snapshot.requests.avg_queue_wait_ms == 10.0
        assert snapshot.requests.avg_execution_ms == 20.0
        assert snapshot.requests.avg_total_latency_ms == 30.0
        assert snapshot.requests.min_total_latency_ms == 30.0
        assert snapshot.requests.max_total_latency_ms == 30.0
        assert snapshot.requests.p50_total_latency_ms == 30.0
        assert snapshot.throughput.total_input_tokens == 15
        assert snapshot.throughput.total_output_tokens == 25

    def test_record_multiple_requests_mixed_status(self) -> None:
        collector = MetricsCollector()
        # Completed
        collector.record_request(
            RequestMetrics(
                request_id="req-1",
                priority=0,
                queue_wait_ms=5.0,
                execution_ms=20.0,
                total_latency_ms=25.0,
                status=RequestStatus.COMPLETED,
                input_tokens=10,
                output_tokens=10,
                max_tokens=32,
                backend_name="mock",
            )
        )
        # Failed
        collector.record_request(
            RequestMetrics(
                request_id="req-2",
                priority=0,
                queue_wait_ms=10.0,
                execution_ms=15.0,
                total_latency_ms=25.0,
                status=RequestStatus.FAILED,
                max_tokens=32,
                backend_name="mock",
                error_message="Backend timeout",
            )
        )
        # Cancelled
        collector.record_request(
            RequestMetrics(
                request_id="req-3",
                priority=0,
                queue_wait_ms=15.0,
                execution_ms=0.0,
                total_latency_ms=15.0,
                status=RequestStatus.CANCELLED,
                max_tokens=32,
                backend_name="mock",
            )
        )

        snapshot = collector.snapshot()
        assert snapshot.requests.total_requests == 3
        assert snapshot.requests.completed_requests == 1
        assert snapshot.requests.failed_requests == 1
        assert snapshot.requests.cancelled_requests == 1
        assert snapshot.requests.avg_queue_wait_ms == 10.0  # (5+10+15)/3
        assert snapshot.requests.min_total_latency_ms == 15.0
        assert snapshot.requests.max_total_latency_ms == 25.0

    def test_record_batches(self) -> None:
        collector = MetricsCollector()
        collector.record_batch(
            BatchMetrics(
                batch_id="b-1",
                size=2,
                batch_formation_wait_ms=5.0,
                execution_ms=30.0,
                total_max_tokens=64,
                backend_name="mock",
                request_ids=("r1", "r2"),
                completed_request_count=2,
                failed_request_count=0,
            )
        )
        collector.record_batch(
            BatchMetrics(
                batch_id="b-2",
                size=4,
                batch_formation_wait_ms=15.0,
                execution_ms=50.0,
                total_max_tokens=128,
                backend_name="mock",
                request_ids=("r3", "r4", "r5", "r6"),
                completed_request_count=3,
                failed_request_count=1,
            )
        )

        snapshot = collector.snapshot()
        assert snapshot.batches.total_batches == 2
        assert snapshot.batches.completed_batches == 1
        assert snapshot.batches.failed_batches == 1
        assert snapshot.batches.avg_batch_size == 3.0
        assert snapshot.batches.min_batch_size == 2
        assert snapshot.batches.max_batch_size == 4
        assert snapshot.batches.avg_batch_formation_wait_ms == 10.0
        assert snapshot.batches.avg_batch_execution_ms == 40.0

    def test_queue_and_concurrency_tracking(self) -> None:
        collector = MetricsCollector()
        collector.record_queue_depth(5)
        collector.record_queue_depth(12)
        collector.record_queue_depth(3)

        collector.record_active_concurrency(2)
        collector.record_active_concurrency(4)
        collector.record_active_concurrency(1)

        snapshot = collector.snapshot()
        assert snapshot.queue.current_queue_depth == 3
        assert snapshot.queue.peak_queue_depth == 12
        assert snapshot.queue.current_active_requests == 1
        assert snapshot.queue.peak_active_requests == 4

    def test_recent_metrics_and_reset(self) -> None:
        collector = MetricsCollector()
        for i in range(5):
            collector.record_request(
                RequestMetrics(
                    request_id=f"r-{i}",
                    priority=0,
                    queue_wait_ms=float(i),
                    execution_ms=10.0,
                    total_latency_ms=10.0 + i,
                    status=RequestStatus.COMPLETED,
                    max_tokens=16,
                    backend_name="mock",
                )
            )

        recent = collector.get_recent_requests(limit=3)
        assert len(recent) == 3
        assert recent[-1].request_id == "r-4"
        assert collector.total_recorded_requests == 5

        collector.reset()
        assert collector.total_recorded_requests == 0
        assert collector.total_recorded_batches == 0
        snapshot = collector.snapshot()
        assert snapshot.requests.total_requests == 0
        assert snapshot.queue.peak_queue_depth == 0

    def test_max_history_bounding(self) -> None:
        collector = MetricsCollector(max_history=5)
        for i in range(10):
            collector.record_request(
                RequestMetrics(
                    request_id=f"req-{i}",
                    priority=0,
                    queue_wait_ms=1.0,
                    execution_ms=2.0,
                    total_latency_ms=3.0,
                    status=RequestStatus.COMPLETED,
                    max_tokens=16,
                    backend_name="mock",
                )
            )
        assert collector.total_recorded_requests == 5
        recent = collector.get_recent_requests()
        assert [r.request_id for r in recent] == [f"req-{i}" for i in range(5, 10)]

    def test_thread_safety(self) -> None:
        collector = MetricsCollector()
        num_threads = 8
        items_per_thread = 50

        def worker(tid: int) -> None:
            for i in range(items_per_thread):
                collector.record_request(
                    RequestMetrics(
                        request_id=f"t{tid}-{i}",
                        priority=0,
                        queue_wait_ms=1.0,
                        execution_ms=2.0,
                        total_latency_ms=3.0,
                        status=RequestStatus.COMPLETED,
                        max_tokens=16,
                        backend_name="mock",
                    )
                )
                collector.record_queue_depth(i)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert collector.total_recorded_requests == num_threads * items_per_thread
        snapshot = collector.snapshot()
        assert snapshot.requests.total_requests == num_threads * items_per_thread
