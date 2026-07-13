"""Orchestrate the PET/wetbulb pipeline: pull ERA5, compute, and load Postgres.

Replaces the pull_all.sh bash orchestrator. Configuration comes from CLI flags
with the historical pull_all.sh environment variables (MODE, YEARS, PRODUCTS,
USE_CLOUD_RUN, ...) honored as defaults, so existing invocations keep working.

Loads are staged upserts on (location_id, date): new data lands on top of the
existing history, which removes the old export-history/merge-CSV steps and
makes any run safe to repeat.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, LiteralString, cast

import psycopg
from dotenv import load_dotenv
from tqdm.auto import tqdm

import cities as cities_module
import locations as locations_module
from load import (
    COPY_BATCH_SIZE,
    DEFAULT_LOAD_WORKERS,
    _load_table_files,
    bulk_insert_csv_files,
    execute_sql_file,
    refresh_query_planner_statistics,
)
from refresh_views import refresh_materialized_views
from shared_config import DATABASE_CONFIG_HINT, resolve_database_uri

if TYPE_CHECKING:
    from psycopg import Connection

LOGGER = logging.getLogger("pipeline")

ERA5_START_YEAR = 2000
MIN_PET_YEARS_FOR_FORECAST = 10
SMOKE_BATCH_HOURS = 168
SMOKE_TIME_SHARD_COUNT = 53
SMOKE_TIME_SHARD_INDEX = 17
FULL_BATCH_HOURS = 720
FULL_TIME_SHARD_COUNT = 13
PRODUCT_TABLES = ("pet", "wetbulb")

# nldas-worker downloads hourly HTTP granules rather than reading a GCS zarr
# store, but reuses the same batch/time-shard partitioning defaults as the
# ERA5 worker since both are tuned for similar per-task run times.
NLDAS_DEFAULT_DOWNLOAD_WORKERS = 16

# Keep numpy/BLAS single-threaded inside worker subprocesses; parallelism is
# managed at the process level.
SINGLE_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAX_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

_ACTIVE_PROCESSES: set[subprocess.Popen[bytes]] = set()
_PROCESS_LOCK = threading.Lock()


class PipelineError(RuntimeError):
    """Raised when a pipeline step fails."""


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    return default if raw_value in (None, "") else int(cast("str", raw_value))


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value == "1"


@dataclass
class PipelineConfig:
    """Resolved pipeline settings."""

    mode: str
    years: list[int]
    products: str
    use_cloud_run: bool
    skip_era5_pull: bool
    skip_db_load: bool
    provision_cloud_run: bool
    sync_from_s3: bool
    skip_remote_clear: bool
    truncate: bool
    views: str
    months: list[int]
    gcp_project: str
    gcp_region: str
    cloud_run_job: str
    nldas_cloud_run_job: str
    s3_bucket: str
    s3_prefix: str
    clear_max_workers: int
    batch_hours: int
    time_shard_count: int
    time_shard_indexes: list[int]
    nldas_batch_hours: int
    nldas_time_shard_count: int
    nldas_time_shard_indexes: list[int]
    download_workers: int
    city_shard_count: int
    job_limit: int
    load_workers: int
    concurrency_profile: str
    batch_workers: int

    @property
    def do_pet(self) -> bool:
        """Whether the PET product is selected."""
        return self.products in ("pet", "both")

    @property
    def do_wetbulb(self) -> bool:
        """Whether the wetbulb product is selected."""
        return self.products in ("wetbulb", "both")

    @property
    def selected_products(self) -> list[str]:
        """Selected product table names."""
        return [p for p in PRODUCT_TABLES if self.products in (p, "both")]

    @property
    def remote_out_dir(self) -> str:
        """S3 URI Cloud Run workers write to."""
        return f"s3://{self.s3_bucket}/{self.s3_prefix}"

    def remote_product_prefix(self, product: str) -> str:
        """S3 key prefix for one product's parquet tree."""
        return f"{self.s3_prefix}/{product}_data_csv"


def _default_years() -> list[int]:
    raw_years = _env_str("YEARS", "").split()
    if raw_years:
        return [int(year) for year in raw_years]
    end_year = datetime.now(tz=UTC).year - 1
    if end_year < ERA5_START_YEAR:
        msg = f"Computed default end year {end_year} is earlier than {ERA5_START_YEAR}; pass --years."
        raise PipelineError(msg)
    return list(range(ERA5_START_YEAR, end_year + 1))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["full", "smoke"],
        default=_env_str("MODE", "full"),
    )
    parser.add_argument("--years", type=int, nargs="+", default=None)
    parser.add_argument(
        "--months",
        type=int,
        nargs="+",
        default=None,
        help="Only process specific months (1-12).",
    )
    parser.add_argument(
        "--products",
        choices=["pet", "wetbulb", "both"],
        default=_env_str("PRODUCTS", "both"),
    )
    parser.add_argument(
        "--skip-era5-pull",
        action="store_true",
        default=_env_flag("SKIP_ERA5_PULL"),
    )
    parser.add_argument(
        "--skip-db-load",
        action="store_true",
        default=_env_flag("SKIP_DB_LOAD"),
    )
    parser.add_argument(
        "--local",
        action="store_true",
        default=not _env_flag("USE_CLOUD_RUN", default=True),
        help="Run google_era5.py locally instead of the Cloud Run worker.",
    )
    parser.add_argument(
        "--provision-cloud-run",
        action="store_true",
        default=_env_flag("PROVISION_CLOUD_RUN"),
    )
    parser.add_argument(
        "--sync-from-s3",
        action="store_true",
        default=_env_flag("SYNC_PET_FROM_S3"),
        help="Sync existing S3 output locally even when the pull is skipped.",
    )
    parser.add_argument(
        "--skip-remote-clear",
        action="store_true",
        default=_env_flag("SKIP_REMOTE_CLEAR"),
    )
    parser.add_argument(
        "--truncate",
        action="store_true",
        help=(
            "Full rebuild: truncate destination tables before loading. "
            "Default is an idempotent upsert on top of existing history."
        ),
    )
    parser.add_argument(
        "--views",
        choices=["recreate", "refresh", "skip"],
        default="recreate",
        help=(
            "recreate: drop and re-run create_views.sql (applies definition "
            "changes; views are down during the load). refresh: keep views "
            "in place and REFRESH MATERIALIZED VIEW afterwards. skip: leave "
            "views untouched."
        ),
    )
    parser.add_argument("--gcp-project", default=_env_str("GCP_PROJECT", ""))
    parser.add_argument("--gcp-region", default=_env_str("GCP_REGION", "us-east1"))
    parser.add_argument(
        "--cloud-run-job",
        default=_env_str("CLOUD_RUN_JOB", "era5-worker"),
    )
    parser.add_argument(
        "--nldas-cloud-run-job",
        default=_env_str("NLDAS_CLOUD_RUN_JOB", "nldas-worker"),
    )
    parser.add_argument(
        "--s3-bucket", default=_env_str("S3_BUCKET", "pet-parquet-files")
    )
    parser.add_argument(
        "--s3-prefix",
        default=_env_str(
            "S3_PREFIX", f"local-run/{os.environ.get('USER', 'unknown-user')}"
        ),
    )
    parser.add_argument(
        "--clear-max-workers",
        type=int,
        default=_env_int("CLEAR_MAX_WORKERS", 8),
    )
    parser.add_argument("--era5-batch-hours", type=int, default=None)
    parser.add_argument("--era5-time-shard-count", type=int, default=None)
    parser.add_argument("--era5-city-shard-count", type=int, default=None)
    parser.add_argument(
        "--era5-job-limit",
        type=int,
        default=None,
        help="Concurrent local ERA5 worker processes (local mode only).",
    )
    parser.add_argument("--nldas-batch-hours", type=int, default=None)
    parser.add_argument("--nldas-time-shard-count", type=int, default=None)
    parser.add_argument(
        "--download-workers",
        type=int,
        default=_env_int("NLDAS_DOWNLOAD_WORKERS", NLDAS_DEFAULT_DOWNLOAD_WORKERS),
        help="Concurrent per-batch NLDAS-2 granule downloads.",
    )
    parser.add_argument(
        "--load-workers",
        type=int,
        default=_env_int("LOAD_WORKERS", DEFAULT_LOAD_WORKERS),
    )
    return parser.parse_args(argv)


def _cpu_count() -> int:
    return os.cpu_count() or 1


def _resolve_shard_config(
    *,
    smoke: bool,
    cli_batch_hours: int | None,
    cli_time_shard_count: int | None,
    batch_hours_env: str,
    time_shard_count_env: str,
    smoke_time_shard_index_env: str,
) -> tuple[int, int, list[int]]:
    """Resolve (batch_hours, time_shard_count, time_shard_indexes) for a worker."""
    batch_hours = cli_batch_hours or _env_int(
        batch_hours_env,
        SMOKE_BATCH_HOURS if smoke else FULL_BATCH_HOURS,
    )
    time_shard_count = cli_time_shard_count or _env_int(
        time_shard_count_env,
        SMOKE_TIME_SHARD_COUNT if smoke else FULL_TIME_SHARD_COUNT,
    )
    if smoke:
        time_shard_indexes = [
            _env_int(smoke_time_shard_index_env, SMOKE_TIME_SHARD_INDEX)
        ]
    else:
        time_shard_indexes = list(range(time_shard_count))
    return batch_hours, time_shard_count, time_shard_indexes


def _resolve_worker_sizing(
    args: argparse.Namespace,
    *,
    use_cloud_run: bool,
    smoke: bool,
) -> tuple[int, int, str, int]:
    """Resolve (city_shard_count, job_limit, concurrency_profile, batch_workers)."""
    if use_cloud_run:
        city_shard_count = args.era5_city_shard_count or _env_int(
            "ERA5_CITY_SHARD_COUNT",
            1,
        )
        job_limit = args.era5_job_limit or _env_int("ERA5_JOB_LIMIT", 1)
        concurrency_profile = _env_str("ERA5_CONCURRENCY_PROFILE", "aggressive")
        batch_workers = _env_int("ERA5_BATCH_WORKERS", 1)
        return city_shard_count, job_limit, concurrency_profile, batch_workers

    available_cpus = max(1, _cpu_count() - 1)
    city_shard_count = args.era5_city_shard_count or _env_int(
        "ERA5_CITY_SHARD_COUNT",
        2,
    )
    job_limit = args.era5_job_limit or _env_int(
        "ERA5_JOB_LIMIT",
        city_shard_count,
    )
    concurrency_profile = _env_str(
        "ERA5_CONCURRENCY_PROFILE",
        "aggressive" if smoke else "balanced",
    )
    batch_workers = _env_int(
        "ERA5_BATCH_WORKERS",
        min(2, max(1, available_cpus // max(1, city_shard_count))) if smoke else 1,
    )
    return city_shard_count, job_limit, concurrency_profile, batch_workers


def build_config(args: argparse.Namespace) -> PipelineConfig:
    """Resolve CLI arguments and environment defaults into a config."""
    use_cloud_run = not args.local
    smoke = args.mode == "smoke"

    batch_hours, time_shard_count, time_shard_indexes = _resolve_shard_config(
        smoke=smoke,
        cli_batch_hours=args.era5_batch_hours,
        cli_time_shard_count=args.era5_time_shard_count,
        batch_hours_env="ERA5_BATCH_HOURS",
        time_shard_count_env="ERA5_TIME_SHARD_COUNT",
        smoke_time_shard_index_env="SMOKE_TIME_SHARD_INDEX",
    )
    nldas_batch_hours, nldas_time_shard_count, nldas_time_shard_indexes = (
        _resolve_shard_config(
            smoke=smoke,
            cli_batch_hours=args.nldas_batch_hours,
            cli_time_shard_count=args.nldas_time_shard_count,
            batch_hours_env="NLDAS_BATCH_HOURS",
            time_shard_count_env="NLDAS_TIME_SHARD_COUNT",
            smoke_time_shard_index_env="NLDAS_SMOKE_TIME_SHARD_INDEX",
        )
    )
    city_shard_count, job_limit, concurrency_profile, batch_workers = (
        _resolve_worker_sizing(args, use_cloud_run=use_cloud_run, smoke=smoke)
    )

    s3_prefix = args.s3_prefix.strip("/")
    if use_cloud_run and not s3_prefix:
        msg = "--s3-prefix must not be empty when using Cloud Run."
        raise PipelineError(msg)

    return PipelineConfig(
        mode=args.mode,
        years=args.years or _default_years(),
        products=args.products,
        use_cloud_run=use_cloud_run,
        skip_era5_pull=args.skip_era5_pull,
        skip_db_load=args.skip_db_load,
        provision_cloud_run=args.provision_cloud_run,
        sync_from_s3=args.sync_from_s3,
        skip_remote_clear=args.skip_remote_clear,
        truncate=args.truncate,
        views=args.views,
        months=args.months or [],
        gcp_project=args.gcp_project,
        gcp_region=args.gcp_region,
        cloud_run_job=args.cloud_run_job,
        nldas_cloud_run_job=args.nldas_cloud_run_job,
        s3_bucket=args.s3_bucket,
        s3_prefix=s3_prefix,
        clear_max_workers=args.clear_max_workers,
        batch_hours=batch_hours,
        time_shard_count=time_shard_count,
        time_shard_indexes=time_shard_indexes,
        nldas_batch_hours=nldas_batch_hours,
        nldas_time_shard_count=nldas_time_shard_count,
        nldas_time_shard_indexes=nldas_time_shard_indexes,
        download_workers=max(1, args.download_workers),
        city_shard_count=max(1, city_shard_count),
        job_limit=max(1, job_limit),
        load_workers=max(1, args.load_workers),
        concurrency_profile=concurrency_profile,
        batch_workers=max(1, batch_workers),
    )


def _run_command(
    command: list[str],
    *,
    label: str,
    extra_env: dict[str, str] | None = None,
) -> None:
    """Run a subprocess, tracking it so interrupts can terminate children."""
    env = None
    if extra_env:
        env = {**os.environ, **extra_env}
    # S603: commands are built from config, not untrusted input.
    process = subprocess.Popen(command, env=env)  # noqa: S603
    with _PROCESS_LOCK:
        _ACTIVE_PROCESSES.add(process)
    try:
        return_code = process.wait()
    finally:
        with _PROCESS_LOCK:
            _ACTIVE_PROCESSES.discard(process)
    if return_code != 0:
        msg = f"{label} failed with exit code {return_code}."
        raise PipelineError(msg)


def _terminate_active_processes() -> None:
    with _PROCESS_LOCK:
        processes = list(_ACTIVE_PROCESSES)
    for process in processes:
        process.terminate()
    for process in processes:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


def _python_command(script: str, *script_args: str) -> list[str]:
    return [sys.executable, script, *script_args]


def prepare_locations(cfg: PipelineConfig) -> int:
    """Generate cities.csv and the derived locations.csv; return city count."""
    LOGGER.info("====== Step 1: Generate cities and locations ======")
    for product in cfg.selected_products:
        Path(f"{product}_data_csv").mkdir(exist_ok=True)
    cities_module.main()
    locations_module.main()

    with Path("cities.csv").open(encoding="utf-8") as cities_csv:
        city_count = sum(1 for _ in cities_csv) - 1
    if city_count < 1:
        msg = "cities.py produced zero locations; cannot continue."
        raise PipelineError(msg)
    return city_count


def _clear_local_year_outputs(cfg: PipelineConfig, products: list[str]) -> None:
    """Remove local parquet for the selected years so stale shards never load."""
    for year in cfg.years:
        for product in products:
            year_dir = Path(f"{product}_data_csv/year={year}")
            if year_dir.exists():
                shutil.rmtree(year_dir)


def _cancel_cloud_run_executions(cfg: PipelineConfig, job: str) -> None:
    command = _python_command(
        "cancel_cloud_run_job_executions.py",
        "--job",
        job,
        "--region",
        cfg.gcp_region,
    )
    if cfg.gcp_project:
        command += ["--project", cfg.gcp_project]
    _run_command(command, label=f"Cancel stale Cloud Run executions ({job})")


def _clear_remote_prefixes(cfg: PipelineConfig, products: list[str]) -> None:
    for product in products:
        prefix = f"{cfg.remote_product_prefix(product)}/"
        LOGGER.info("Clearing s3://%s/%s", cfg.s3_bucket, prefix)
        _run_command(
            _python_command(
                "clear_s3_prefix.py",
                "--bucket",
                cfg.s3_bucket,
                "--prefix",
                prefix,
                "--max-workers",
                str(cfg.clear_max_workers),
            ),
            label=f"Clear remote {product} prefix",
        )


def _maybe_provision_cloud_run(cfg: PipelineConfig) -> None:
    if cfg.use_cloud_run and cfg.provision_cloud_run:
        _run_command(
            ["/bin/bash", "./cloudrun_provision.sh"],
            label="Provision Cloud Run workers",
        )


def _run_local_jobs(
    jobs: list[list[str]],
    cfg: PipelineConfig,
    *,
    desc: str,
    product_label: str,
) -> None:
    LOGGER.info(
        "Running %d local %s job(s) with up to %d concurrent process(es).",
        len(jobs),
        product_label,
        cfg.job_limit,
    )
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=cfg.job_limit) as executor:
        future_labels = {
            executor.submit(
                _run_command,
                job,
                label=" ".join(job[2:8]),
                extra_env=SINGLE_THREAD_ENV,
            ): " ".join(job[2:8])
            for job in jobs
        }
        for future in tqdm(
            as_completed(future_labels),
            total=len(future_labels),
            desc=desc,
        ):
            try:
                future.result()
            except PipelineError as error:
                failures.append(str(error))
    if failures:
        msg = f"{len(failures)} local {product_label} job(s) failed: {failures[:3]}"
        raise PipelineError(msg)


def _cloud_run_worker_args(cfg: PipelineConfig, year: int) -> str:
    worker_args = [
        f"--year={year}",
        f"--city-shard-count={cfg.city_shard_count}",
        f"--time-shard-count={cfg.time_shard_count}",
        f"--max-workers={cfg.batch_workers}",
        f"--batch-hours={cfg.batch_hours}",
        f"--concurrency-profile={cfg.concurrency_profile}",
        "--products=pet",
        f"--out-dir={cfg.remote_out_dir}",
    ]
    if cfg.mode == "smoke":
        worker_args.append(f"--time-shard-index={cfg.time_shard_indexes[0]}")
    if cfg.months:
        worker_args.append("--months")
        worker_args.extend(str(month) for month in cfg.months)
    return ",".join(worker_args)


def _run_cloud_run_year(cfg: PipelineConfig, year: int) -> None:
    """Run one Cloud Run execution per year, fanning out over task indexes.

    In full mode the execution runs with --tasks=<time shards> and each task
    derives its shard from CLOUD_RUN_TASK_INDEX; smoke mode pins a single
    shard explicitly and runs one task.
    """
    tasks = 1 if cfg.mode == "smoke" else cfg.time_shard_count
    command = [
        "gcloud",
        "run",
        "jobs",
        "execute",
        cfg.cloud_run_job,
        "--region",
        cfg.gcp_region,
        "--wait",
        "--tasks",
        str(tasks),
        f"--args={_cloud_run_worker_args(cfg, year)}",
    ]
    if cfg.gcp_project:
        command += ["--project", cfg.gcp_project]
    _run_command(command, label=f"Cloud Run ERA5->PET year={year}")


def _local_era5_jobs(cfg: PipelineConfig) -> list[list[str]]:
    jobs: list[list[str]] = []
    for year in cfg.years:
        for city_shard in range(cfg.city_shard_count):
            for time_shard in cfg.time_shard_indexes:
                script_args = [
                    "--year",
                    str(year),
                    "--city-shard-index",
                    str(city_shard),
                    "--city-shard-count",
                    str(cfg.city_shard_count),
                    "--time-shard-index",
                    str(time_shard),
                    "--time-shard-count",
                    str(cfg.time_shard_count),
                    "--batch-hours",
                    str(cfg.batch_hours),
                    "--max-workers",
                    str(cfg.batch_workers),
                    "--concurrency-profile",
                    cfg.concurrency_profile,
                    "--products",
                    "pet",
                    "--out-dir",
                    ".",
                ]
                if cfg.months:
                    script_args.append("--months")
                    script_args.extend(str(month) for month in cfg.months)
                jobs.append(_python_command("google_era5.py", *script_args))
    return jobs


def run_era5_pull(cfg: PipelineConfig) -> None:
    """Compute ERA5 -> PET parquet, remotely or locally."""
    LOGGER.info("====== Step 2a: Compute ERA5 PET ======")
    _clear_local_year_outputs(cfg, ["pet"])

    if cfg.use_cloud_run:
        _cancel_cloud_run_executions(cfg, cfg.cloud_run_job)
        if not cfg.skip_remote_clear:
            _clear_remote_prefixes(cfg, ["pet"])
        for year in tqdm(cfg.years, desc="Cloud Run PET years"):
            _run_cloud_run_year(cfg, year)
        return

    _run_local_jobs(
        _local_era5_jobs(cfg),
        cfg,
        desc="Local ERA5 jobs",
        product_label="ERA5",
    )


def _nldas_worker_args(cfg: PipelineConfig, year: int) -> str:
    worker_args = [
        f"--year={year}",
        f"--city-shard-count={cfg.city_shard_count}",
        f"--time-shard-count={cfg.nldas_time_shard_count}",
        f"--download-workers={cfg.download_workers}",
        f"--batch-hours={cfg.nldas_batch_hours}",
        f"--out-dir={cfg.remote_out_dir}",
    ]
    if cfg.mode == "smoke":
        worker_args.append(f"--time-shard-index={cfg.nldas_time_shard_indexes[0]}")
    if cfg.months:
        worker_args.append("--months")
        worker_args.extend(str(month) for month in cfg.months)
    return ",".join(worker_args)


def _run_nldas_cloud_run_year(cfg: PipelineConfig, year: int) -> None:
    """Run one Cloud Run execution per year against the nldas-worker job."""
    tasks = 1 if cfg.mode == "smoke" else cfg.nldas_time_shard_count
    command = [
        "gcloud",
        "run",
        "jobs",
        "execute",
        cfg.nldas_cloud_run_job,
        "--region",
        cfg.gcp_region,
        "--wait",
        "--tasks",
        str(tasks),
        f"--args={_nldas_worker_args(cfg, year)}",
    ]
    if cfg.gcp_project:
        command += ["--project", cfg.gcp_project]
    _run_command(command, label=f"Cloud Run NLDAS->wetbulb year={year}")


def _local_nldas_jobs(cfg: PipelineConfig) -> list[list[str]]:
    jobs: list[list[str]] = []
    for year in cfg.years:
        for city_shard in range(cfg.city_shard_count):
            for time_shard in cfg.nldas_time_shard_indexes:
                script_args = [
                    "--year",
                    str(year),
                    "--city-shard-index",
                    str(city_shard),
                    "--city-shard-count",
                    str(cfg.city_shard_count),
                    "--time-shard-index",
                    str(time_shard),
                    "--time-shard-count",
                    str(cfg.nldas_time_shard_count),
                    "--batch-hours",
                    str(cfg.nldas_batch_hours),
                    "--download-workers",
                    str(cfg.download_workers),
                    "--out-dir",
                    ".",
                ]
                if cfg.months:
                    script_args.append("--months")
                    script_args.extend(str(month) for month in cfg.months)
                jobs.append(_python_command("nldas.py", *script_args))
    return jobs


def run_nldas_pull(cfg: PipelineConfig) -> None:
    """Compute NLDAS-2 -> wetbulb parquet, remotely or locally."""
    LOGGER.info("====== Step 2b: Compute NLDAS-2 wetbulb ======")
    _clear_local_year_outputs(cfg, ["wetbulb"])

    if cfg.use_cloud_run:
        _cancel_cloud_run_executions(cfg, cfg.nldas_cloud_run_job)
        if not cfg.skip_remote_clear:
            _clear_remote_prefixes(cfg, ["wetbulb"])
        for year in tqdm(cfg.years, desc="Cloud Run NLDAS years"):
            _run_nldas_cloud_run_year(cfg, year)
        return

    _run_local_jobs(
        _local_nldas_jobs(cfg),
        cfg,
        desc="Local NLDAS jobs",
        product_label="NLDAS",
    )


def sync_outputs_from_s3(cfg: PipelineConfig) -> None:
    """Mirror the remote parquet trees for the selected products locally."""
    for product in cfg.selected_products:
        remote_uri = f"s3://{cfg.s3_bucket}/{cfg.remote_product_prefix(product)}/"
        LOGGER.info("Syncing %s", remote_uri)
        local_dir = Path(f"{product}_data_csv")
        local_dir.mkdir(exist_ok=True)
        _run_command(
            ["aws", "s3", "sync", remote_uri, f"./{local_dir}/", "--delete"],
            label=f"Sync {product} output from S3",
        )


def _discover_product_paths(product: str) -> list[Path]:
    return sorted(Path(f"{product}_data_csv").rglob(f"{product}_batch_*.parquet"))


def assert_outputs_available(cfg: PipelineConfig) -> None:
    """Fail fast when a selected product produced no parquet output."""
    for product in cfg.selected_products:
        if not _discover_product_paths(product):
            msg = f"No {product} batch parquet files found under ./{product}_data_csv."
            raise PipelineError(msg)


def _distinct_year_count(conn: Connection[Any], table: str) -> int:
    query = f"SELECT count(DISTINCT date_part('year', date)) FROM public.{table}"  # noqa: S608 - table names come from PRODUCT_TABLES
    with conn.cursor() as cur:
        cur.execute(cast("LiteralString", query))
        row = cur.fetchone()
    return int(row[0]) if row else 0


def _check_history_depth(conn: Connection[Any], cfg: PipelineConfig) -> None:
    """Enforce the forecast history requirement before building views."""
    if cfg.do_pet:
        pet_years = _distinct_year_count(conn, "pet")
        if pet_years < MIN_PET_YEARS_FOR_FORECAST:
            msg = (
                f"Need at least {MIN_PET_YEARS_FOR_FORECAST} PET years to "
                f"derive pet_forecast; found {pet_years}. Backfill more "
                "history or run with --views skip."
            )
            raise PipelineError(msg)
    if cfg.do_wetbulb:
        wetbulb_years = _distinct_year_count(conn, "wetbulb")
        if wetbulb_years < MIN_PET_YEARS_FOR_FORECAST:
            LOGGER.warning(
                "Only %d wetbulb year(s) present; wetbulb forecasts will be "
                "empty until more history is backfilled.",
                wetbulb_years,
            )


def run_db_load(cfg: PipelineConfig) -> None:
    """Load locations and the selected products, then rebuild/refresh views."""
    db_uri = resolve_database_uri()
    if not db_uri:
        LOGGER.warning(
            "Postgres database credentials are not configured. %s Skipping DB load.",
            DATABASE_CONFIG_HINT,
        )
        return

    LOGGER.info("====== Step 3: Load data into Postgres ======")
    conn: Connection[Any] = psycopg.connect(db_uri)
    conn.autocommit = True
    try:
        if cfg.views == "recreate":
            execute_sql_file(conn, "drop_views.sql")
        execute_sql_file(conn, "create_tables.sql")

        bulk_insert_csv_files(
            conn,
            [Path("locations.csv")],
            "locations",
            batch_size=COPY_BATCH_SIZE,
            truncate=cfg.truncate,
        )

        for product in cfg.selected_products:
            _load_table_files(
                conn,
                db_uri,
                _discover_product_paths(product),
                product,
                batch_size=COPY_BATCH_SIZE,
                truncate=cfg.truncate,
                workers=cfg.load_workers,
            )

        if cfg.views == "skip":
            refresh_query_planner_statistics(conn)
            return

        _check_history_depth(conn, cfg)
        LOGGER.info("====== Step 4: %s views ======", cfg.views.capitalize())
        if cfg.views == "recreate":
            execute_sql_file(conn, "create_views.sql")
        else:
            refresh_materialized_views(conn)
        refresh_query_planner_statistics(conn)
    finally:
        conn.close()
        LOGGER.info("Database connection closed.")


def run_pipeline(cfg: PipelineConfig) -> None:
    """Execute the configured pipeline stages in order."""
    LOGGER.info(
        "Pipeline mode=%s products=%s years=%s via %s",
        cfg.mode,
        cfg.products,
        f"{cfg.years[0]}-{cfg.years[-1]}" if len(cfg.years) > 1 else cfg.years[0],
        "Cloud Run" if cfg.use_cloud_run else "local compute",
    )
    prepare_locations(cfg)

    if not cfg.skip_era5_pull:
        _maybe_provision_cloud_run(cfg)
        if cfg.do_pet:
            run_era5_pull(cfg)
        if cfg.do_wetbulb:
            run_nldas_pull(cfg)
        if cfg.use_cloud_run:
            sync_outputs_from_s3(cfg)
    elif cfg.use_cloud_run and cfg.sync_from_s3:
        sync_outputs_from_s3(cfg)

    assert_outputs_available(cfg)

    if cfg.skip_db_load:
        LOGGER.info("====== Steps 3-4: Skipping DB load ======")
    else:
        run_db_load(cfg)
    LOGGER.info("====== Pipeline complete! ======")


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    load_dotenv(override=False)
    exit_code = 0
    try:
        cfg = build_config(_parse_args(argv))
        run_pipeline(cfg)
    except KeyboardInterrupt:
        LOGGER.warning("Pipeline interrupted; terminating child processes...")
        _terminate_active_processes()
        exit_code = 130
    except PipelineError:
        LOGGER.exception("Pipeline failed.")
        _terminate_active_processes()
        exit_code = 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
