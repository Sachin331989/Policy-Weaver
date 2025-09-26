"""Fabric Capacity Stress Test
================================

This module coordinates synthetic Spark workloads to stress-test a Microsoft
Fabric capacity (for example an F64 SKU) and observe bursting/smoothing
behaviour.  The entry point can be executed from a Fabric Spark notebook or via
``spark-submit`` in an environment that already has access to a Fabric capacity.

Prerequisites
-------------
* A Microsoft Fabric workspace that is attached to the target capacity.
* A PySpark runtime (Fabric Spark runtime or a local pyspark installation).
* Optional: The ``requests`` package and an ``FABRIC_API_TOKEN`` environment
  variable to collect capacity metrics through the Fabric REST API.

Typical usage within Fabric:

```
spark-submit fabric_capacity_stress_test.py --mode both --duration 600 \
    --burst-pattern "64:120,16:60" --job-size 500000000 --concurrency 32
```

The script records job level metrics such as latency, throughput, and whether
throttling indicators were observed.  Optionally it can poll the Fabric REST API
for capacity level metrics when an API token is provided.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Sequence, Tuple

try:
    # PySpark provides the distributed DataFrame operations that power the
    # synthetic workloads used in the stress tests.
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
except ImportError as exc:  # pragma: no cover - pyspark is required at runtime
    raise SystemExit(
        "PySpark is required to execute this script. Install pyspark or run "
        "within a Microsoft Fabric Spark runtime."
    ) from exc

_LOGGER = logging.getLogger("fabric.stress")


@dataclass
class JobResult:
    """Container for the outcome of a single Spark workload."""

    job_id: str
    workload: str
    start_ts: float
    end_ts: float
    duration_s: float
    rows_processed: int
    status: str
    exception: Optional[str] = None
    throttle_flag: bool = False


@dataclass
class SummaryMetrics:
    """Aggregated metrics across all executed jobs."""

    total_jobs: int
    succeeded: int
    throttled: int
    failed: int
    avg_latency_s: float
    p95_latency_s: float
    throughput_rows_per_s: float
    throughput_jobs_per_s: float


class FabricMetricsClient:
    """Optional client to poll Fabric capacity metrics.

    The client expects an Azure AD access token in ``FABRIC_API_TOKEN`` and uses
    the Fabric REST API (https://learn.microsoft.com/fabric/) to capture
    capacity utilisation.  The collected samples are stored in ``self.samples``.
    """

    def __init__(self, capacity_id: Optional[str] = None, workspace_id: Optional[str] = None):
        self.capacity_id = capacity_id
        self.workspace_id = workspace_id
        self.samples: List[Dict[str, float]] = []
        self._lock = threading.Lock()

        # Access token is supplied via environment variable so the script can be
        # executed non-interactively in notebooks or pipelines.
        token = os.getenv("FABRIC_API_TOKEN")
        if (capacity_id or workspace_id) and not token:
            raise SystemExit(
                "Fabric metrics requested but FABRIC_API_TOKEN is not set."
            )
        self._token = token

        try:
            import requests  # noqa: WPS433 (used only when metrics are enabled)

            self._requests = requests
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise SystemExit(
                "requests must be installed to poll Fabric capacity metrics"
            ) from exc

    def poll(self) -> None:
        """Capture a capacity metric sample if the client is configured."""

        if not (self.capacity_id and self._token):
            return

        headers = {"Authorization": f"Bearer {self._token}"}
        # Fabric API currently exposes capacity metrics via the admin endpoint;
        # we forward samples to ``self.samples`` for later reporting.
        url = (
            "https://api.fabric.microsoft.com/v1/capacities/"
            f"{self.capacity_id}/metrics"
        )
        params = {"workspaceId": self.workspace_id} if self.workspace_id else None
        response = self._requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        timestamp = time.time()

        with self._lock:
            for metric in payload.get("value", []):
                self.samples.append(
                    {
                        "timestamp": timestamp,
                        "metricName": metric.get("name"),
                        "metricValue": metric.get("value"),
                        "metricUnit": metric.get("unit"),
                    }
                )


class WorkloadRunner:
    """Run synthetic Spark workloads to exercise a Fabric capacity."""

    def __init__(
        self,
        spark: SparkSession,
        concurrency: int,
        job_size: int,
        partitions: int,
        shuffle_partitions: Optional[int] = None,
        enable_cache: bool = False,
        metrics_client: Optional[FabricMetricsClient] = None,
    ) -> None:
        self.spark = spark
        self.concurrency = max(1, concurrency)
        self.job_size = job_size
        self.partitions = partitions
        self.shuffle_partitions = shuffle_partitions
        self.enable_cache = enable_cache
        self.metrics_client = metrics_client

        if shuffle_partitions:
            self.spark.conf.set("spark.sql.shuffle.partitions", shuffle_partitions)

        _LOGGER.debug(
            "Initialised WorkloadRunner with concurrency=%s job_size=%s partitions=%s",
            concurrency,
            job_size,
            partitions,
        )

    # ------------------------------------------------------------------
    # Public orchestration helpers
    # ------------------------------------------------------------------
    def run_sustained(self, duration_s: int, label: str = "sustained") -> List[JobResult]:
        """Generate a sustained load for ``duration_s`` seconds."""

        _LOGGER.info("Starting sustained workload for %s seconds", duration_s)
        return self._run_phase(
            target_duration_s=duration_s,
            concurrency=self.concurrency,
            label=label,
        )

    def run_spiky(
        self,
        burst_pattern: Sequence[Tuple[int, int]],
        idle_between_bursts_s: int,
        label: str = "spiky",
    ) -> List[JobResult]:
        """Execute a spiky pattern defined by ``burst_pattern``.

        Args:
            burst_pattern: Iterable of ``(concurrency, duration_seconds)`` tuples.
            idle_between_bursts_s: Seconds to sleep between bursts.
        """

        results: List[JobResult] = []
        for idx, (burst_concurrency, burst_duration) in enumerate(burst_pattern, start=1):
            burst_label = f"{label}-burst-{idx}"
            _LOGGER.info(
                "Starting burst %s with concurrency=%s duration=%s",
                burst_label,
                burst_concurrency,
                burst_duration,
            )
            results.extend(
                self._run_phase(
                    target_duration_s=burst_duration,
                    concurrency=max(1, burst_concurrency),
                    label=burst_label,
                )
            )
            if idle_between_bursts_s > 0 and idx < len(burst_pattern):
                _LOGGER.info("Idling for %s seconds between bursts", idle_between_bursts_s)
                time.sleep(idle_between_bursts_s)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _run_phase(
        self,
        target_duration_s: int,
        concurrency: int,
        label: str,
    ) -> List[JobResult]:
        """Run a phase with a fixed concurrency until ``target_duration_s``."""

        executor = ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix=label)
        end_time = time.time() + target_duration_s
        in_flight: Dict[Future[JobResult], str] = {}
        results: List[JobResult] = []
        job_seq = 0

        def submit_job() -> None:
            nonlocal job_seq
            job_id = f"{label}-{job_seq}"
            future = executor.submit(self._execute_job, job_id, label)
            in_flight[future] = job_id
            job_seq += 1

        for _ in range(concurrency):
            submit_job()

        while in_flight:
            completed, _ = wait(in_flight.keys(), timeout=1, return_when=FIRST_COMPLETED)
            for future in completed:
                job_id = in_flight.pop(future)
                try:
                    result = future.result()
                    _LOGGER.debug("Job %s completed in %.2fs", job_id, result.duration_s)
                    results.append(result)
                except Exception as exc:  # pragma: no cover - defensive
                    _LOGGER.exception("Job %s failed: %s", job_id, exc)
                if time.time() < end_time:
                    submit_job()

        executor.shutdown(wait=True)
        return results

    def _execute_job(self, job_id: str, workload: str) -> JobResult:
        """Run a single Spark job and collect execution metrics."""

        start_ts = time.time()
        throttle_flag = False
        status = "success"
        exception_text = None

        try:
            df = self.spark.range(0, self.job_size, numPartitions=self.partitions)
            heavy_df = (
                df.withColumn("scaled", F.col("id") * F.lit(math.pi))
                .withColumn("sin", F.sin("scaled"))
                .withColumn("cos", F.cos("scaled"))
                .repartition(self.partitions)
            )
            aggregated = (
                heavy_df.groupBy((F.col("id") % 23).alias("bucket"))
                .agg(
                    F.sum("sin").alias("sum_sin"),
                    F.sum("cos").alias("sum_cos"),
                    F.avg("scaled").alias("avg_scaled"),
                )
                .orderBy("bucket")
            )
            if self.enable_cache:
                aggregated.cache()
                aggregated.count()
            result = aggregated.collect()
            rows_processed = len(result) * self.partitions
        except Exception as exc:  # pragma: no cover - runtime defensive handling
            exception_text = str(exc)
            lowered = exception_text.lower()
            if "throttle" in lowered or "rate limit" in lowered:
                throttle_flag = True
                status = "throttled"
            else:
                status = "failed"
            rows_processed = 0
        finally:
            if self.metrics_client:
                try:
                    self.metrics_client.poll()
                except Exception as exc:  # pragma: no cover - do not fail workload
                    _LOGGER.warning("Fabric metrics polling failed: %s", exc)

        end_ts = time.time()
        return JobResult(
            job_id=job_id,
            workload=workload,
            start_ts=start_ts,
            end_ts=end_ts,
            duration_s=end_ts - start_ts,
            rows_processed=rows_processed,
            status=status,
            exception=exception_text,
            throttle_flag=throttle_flag,
        )


# ----------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------

def parse_burst_pattern(pattern: str) -> List[Tuple[int, int]]:
    """Parse burst pattern strings such as ``"64:120,16:30"``."""

    if not pattern:
        return []

    bursts: List[Tuple[int, int]] = []
    for entry in pattern.split(","):
        entry = entry.strip()
        if not entry:
            continue
        try:
            concurrency_str, duration_str = entry.split(":", maxsplit=1)
            bursts.append((int(concurrency_str), int(duration_str)))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "Burst pattern must be a comma separated list of concurrency:duration"
            ) from exc
    return bursts


def calculate_summary(metrics: Sequence[JobResult]) -> SummaryMetrics:
    if not metrics:
        return SummaryMetrics(0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0)

    durations = [m.duration_s for m in metrics]
    sorted_durations = sorted(durations)
    p95_index = min(len(sorted_durations) - 1, int(math.ceil(0.95 * len(sorted_durations))) - 1)

    total_duration = max(m.end_ts for m in metrics) - min(m.start_ts for m in metrics)
    total_duration = max(total_duration, 1e-6)
    total_rows = sum(m.rows_processed for m in metrics)

    throttled = sum(1 for m in metrics if m.throttle_flag)
    failed = sum(1 for m in metrics if m.status == "failed")
    succeeded = sum(1 for m in metrics if m.status == "success")

    return SummaryMetrics(
        total_jobs=len(metrics),
        succeeded=succeeded,
        throttled=throttled,
        failed=failed,
        avg_latency_s=mean(durations),
        p95_latency_s=sorted_durations[p95_index],
        throughput_rows_per_s=total_rows / total_duration,
        throughput_jobs_per_s=len(metrics) / total_duration,
    )


def write_metrics(metrics: Sequence[JobResult], path: Path) -> None:
    payload = [asdict(job) for job in metrics]
    path.write_text(json.dumps(payload, indent=2))
    _LOGGER.info("Wrote %s job metrics to %s", len(metrics), path)


def write_summary(summary: SummaryMetrics, path: Path) -> None:
    path.write_text(json.dumps(asdict(summary), indent=2))
    _LOGGER.info("Summary metrics written to %s", path)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fabric F64 capacity stress tester")
    parser.add_argument(
        "--mode",
        default="both",
        choices=("sustained", "spiky", "both"),
        help="Workload mode to execute",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=600,
        help="Duration in seconds for sustained workloads",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=32,
        help="Number of concurrent jobs during sustained workloads",
    )
    parser.add_argument(
        "--job-size",
        type=int,
        default=250_000_000,
        help="Number of rows to generate per job",
    )
    parser.add_argument(
        "--partitions",
        type=int,
        default=64,
        help="Number of partitions to generate for synthetic data",
    )
    parser.add_argument(
        "--shuffle-partitions",
        type=int,
        default=None,
        help="Optional override for spark.sql.shuffle.partitions",
    )
    parser.add_argument(
        "--enable-cache",
        action="store_true",
        help="Cache aggregated data to increase pressure on the storage subsystem",
    )
    parser.add_argument(
        "--burst-pattern",
        type=parse_burst_pattern,
        default=parse_burst_pattern("64:180,16:120,72:90"),
        help="Comma separated list of concurrency:duration values for spiky workloads",
    )
    parser.add_argument(
        "--idle-between-bursts",
        type=int,
        default=30,
        help="Idle time between bursts in seconds",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("fabric_job_metrics.json"),
        help="File path to write detailed job metrics",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("fabric_summary_metrics.json"),
        help="File path to write aggregated summary metrics",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (e.g. INFO, DEBUG)",
    )
    parser.add_argument(
        "--capacity-id",
        default=None,
        help="Optional Fabric capacity identifier for REST metric polling",
    )
    parser.add_argument(
        "--workspace-id",
        default=None,
        help="Optional Fabric workspace identifier for scoped metrics",
    )
    return parser.parse_args(argv)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    configure_logging(args.log_level)

    spark = SparkSession.builder.appName("fabric-capacity-stress").getOrCreate()

    metrics_client: Optional[FabricMetricsClient] = None
    if args.capacity_id:
        metrics_client = FabricMetricsClient(args.capacity_id, args.workspace_id)

    runner = WorkloadRunner(
        spark=spark,
        concurrency=args.concurrency,
        job_size=args.job_size,
        partitions=args.partitions,
        shuffle_partitions=args.shuffle_partitions,
        enable_cache=args.enable_cache,
        metrics_client=metrics_client,
    )

    all_results: List[JobResult] = []
    if args.mode in {"sustained", "both"}:
        all_results.extend(runner.run_sustained(duration_s=args.duration))
    if args.mode in {"spiky", "both"} and args.burst_pattern:
        all_results.extend(
            runner.run_spiky(
                burst_pattern=args.burst_pattern,
                idle_between_bursts_s=args.idle_between_bursts,
            )
        )

    summary = calculate_summary(all_results)
    write_metrics(all_results, args.metrics_output)
    write_summary(summary, args.summary_output)

    _LOGGER.info("Executed %s jobs", summary.total_jobs)
    _LOGGER.info(
        "Throughput: %.2f jobs/s, %.2f rows/s",
        summary.throughput_jobs_per_s,
        summary.throughput_rows_per_s,
    )
    _LOGGER.info(
        "Latency avg=%.2fs p95=%.2fs (throttled=%s failed=%s)",
        summary.avg_latency_s,
        summary.p95_latency_s,
        summary.throttled,
        summary.failed,
    )

    spark.stop()


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
