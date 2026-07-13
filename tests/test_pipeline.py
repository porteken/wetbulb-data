"""Tests for the pipeline.py orchestrator."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

import pipeline


def _config(argv: list[str]) -> pipeline.PipelineConfig:
    return pipeline.build_config(pipeline._parse_args(argv))


@pytest.fixture(autouse=True)
def _clean_pipeline_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "MODE",
        "YEARS",
        "PRODUCTS",
        "USE_CLOUD_RUN",
        "SKIP_ERA5_PULL",
        "SKIP_DB_LOAD",
        "ERA5_BATCH_HOURS",
        "ERA5_TIME_SHARD_COUNT",
        "ERA5_CITY_SHARD_COUNT",
        "ERA5_JOB_LIMIT",
        "ERA5_CONCURRENCY_PROFILE",
        "ERA5_BATCH_WORKERS",
        "SMOKE_TIME_SHARD_INDEX",
        "S3_PREFIX",
        "LOAD_WORKERS",
    ):
        monkeypatch.delenv(name, raising=False)


class TestBuildConfig:
    def test_full_mode_defaults(self) -> None:
        cfg = _config(["--years", "2024"])

        assert cfg.mode == "full"
        assert cfg.use_cloud_run is True
        assert cfg.batch_hours == pipeline.FULL_BATCH_HOURS
        assert cfg.time_shard_count == pipeline.FULL_TIME_SHARD_COUNT
        assert cfg.time_shard_indexes == list(range(13))
        assert cfg.products == "both"
        assert cfg.selected_products == ["pet", "wetbulb"]
        assert cfg.city_shard_count == 1

    def test_smoke_mode_defaults(self) -> None:
        cfg = _config(["--mode", "smoke", "--years", "2024"])

        assert cfg.batch_hours == pipeline.SMOKE_BATCH_HOURS
        assert cfg.time_shard_count == pipeline.SMOKE_TIME_SHARD_COUNT
        assert cfg.time_shard_indexes == [pipeline.SMOKE_TIME_SHARD_INDEX]

    def test_env_variables_are_honored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MODE", "smoke")
        monkeypatch.setenv("YEARS", "2020 2021")
        monkeypatch.setenv("PRODUCTS", "pet")
        monkeypatch.setenv("USE_CLOUD_RUN", "0")
        monkeypatch.setenv("ERA5_BATCH_HOURS", "24")

        cfg = _config([])

        assert cfg.mode == "smoke"
        assert cfg.years == [2020, 2021]
        assert cfg.products == "pet"
        assert cfg.use_cloud_run is False
        assert cfg.batch_hours == 24
        assert cfg.do_pet is True
        assert cfg.do_wetbulb is False

    def test_cli_flags_override_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PRODUCTS", "pet")

        cfg = _config(["--products", "wetbulb", "--years", "2024"])

        assert cfg.products == "wetbulb"

    def test_empty_s3_prefix_rejected_for_cloud_run(self) -> None:
        with pytest.raises(pipeline.PipelineError, match="s3-prefix"):
            _config(["--years", "2024", "--s3-prefix", "/"])

    def test_default_years_span_2000_to_previous_year(self) -> None:
        cfg = _config([])

        assert cfg.years[0] == pipeline.ERA5_START_YEAR
        assert len(cfg.years) >= 1

    def test_remote_paths(self) -> None:
        cfg = _config(["--years", "2024", "--s3-prefix", "run/x/"])

        assert cfg.remote_out_dir == "s3://pet-parquet-files/run/x"
        assert cfg.remote_product_prefix("pet") == "run/x/pet_data_csv"


class TestWorkerArgs:
    def test_full_mode_omits_time_shard_index(self) -> None:
        cfg = _config(["--years", "2024"])

        args_csv = pipeline._cloud_run_worker_args(cfg, 2024)

        assert "--time-shard-index" not in args_csv
        assert "--year=2024" in args_csv
        assert f"--time-shard-count={pipeline.FULL_TIME_SHARD_COUNT}" in args_csv

    def test_smoke_mode_pins_time_shard_index(self) -> None:
        cfg = _config(["--mode", "smoke", "--years", "2024"])

        args_csv = pipeline._cloud_run_worker_args(cfg, 2024)

        assert f"--time-shard-index={pipeline.SMOKE_TIME_SHARD_INDEX}" in args_csv

    def test_months_are_forwarded(self) -> None:
        cfg = _config(["--years", "2024", "--months", "1", "2"])

        args_csv = pipeline._cloud_run_worker_args(cfg, 2024)

        assert "--months,1,2" in args_csv

    def test_local_jobs_cover_all_shards(self) -> None:
        cfg = _config(
            [
                "--local",
                "--years",
                "2023",
                "2024",
                "--era5-city-shard-count",
                "2",
                "--era5-time-shard-count",
                "3",
            ]
        )
        cfg.time_shard_indexes = list(range(3))

        jobs = pipeline._local_era5_jobs(cfg)

        assert len(jobs) == 2 * 2 * 3
        assert all("google_era5.py" in job[1] for job in jobs)

    def test_era5_worker_always_requests_pet_only(self) -> None:
        """ERA5 no longer writes wetbulb; it must always run --products pet."""
        cfg = _config(["--years", "2024", "--products", "both"])

        args_csv = pipeline._cloud_run_worker_args(cfg, 2024)

        assert "--products=pet" in args_csv
        assert "--products=both" not in args_csv
        assert "--products=wetbulb" not in args_csv

    def test_local_era5_jobs_always_pass_products_pet(self) -> None:
        cfg = _config(["--local", "--years", "2024", "--products", "both"])
        cfg.time_shard_indexes = [0]

        jobs = pipeline._local_era5_jobs(cfg)

        assert all("pet" in job for job in jobs)
        assert all("both" not in job and "wetbulb" not in job for job in jobs)


class TestNldasWorkerArgs:
    def test_full_mode_omits_time_shard_index(self) -> None:
        cfg = _config(["--years", "2024"])

        args_csv = pipeline._nldas_worker_args(cfg, 2024)

        assert "--time-shard-index" not in args_csv
        assert "--year=2024" in args_csv
        assert f"--time-shard-count={pipeline.FULL_TIME_SHARD_COUNT}" in args_csv
        assert (
            f"--download-workers={pipeline.NLDAS_DEFAULT_DOWNLOAD_WORKERS}" in args_csv
        )

    def test_smoke_mode_pins_time_shard_index(self) -> None:
        cfg = _config(["--mode", "smoke", "--years", "2024"])

        args_csv = pipeline._nldas_worker_args(cfg, 2024)

        assert f"--time-shard-index={pipeline.SMOKE_TIME_SHARD_INDEX}" in args_csv

    def test_months_are_forwarded(self) -> None:
        cfg = _config(["--years", "2024", "--months", "1", "2"])

        args_csv = pipeline._nldas_worker_args(cfg, 2024)

        assert "--months,1,2" in args_csv

    def test_local_jobs_cover_all_shards(self) -> None:
        cfg = _config(
            [
                "--local",
                "--years",
                "2023",
                "2024",
                "--era5-city-shard-count",
                "2",
                "--nldas-time-shard-count",
                "3",
            ]
        )
        cfg.nldas_time_shard_indexes = list(range(3))

        jobs = pipeline._local_nldas_jobs(cfg)

        assert len(jobs) == 2 * 2 * 3
        assert all("nldas.py" in job[1] for job in jobs)


class TestRunCommand:
    def test_success(self) -> None:
        pipeline._run_command(["true"], label="ok")

    def test_failure_raises_pipeline_error(self) -> None:
        with pytest.raises(pipeline.PipelineError, match="exit code"):
            pipeline._run_command(["false"], label="boom")


class TestPrepareLocations:
    def test_generates_and_counts_cities(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)

        def fake_cities_main() -> None:
            Path("cities.csv").write_text(
                "location_id,city,state,lat,lng\n0,A,B,1,2\n1,C,D,3,4\n"
            )

        monkeypatch.setattr(pipeline.cities_module, "main", fake_cities_main)
        monkeypatch.setattr(pipeline.locations_module, "main", MagicMock())

        cfg = _config(["--years", "2024"])
        assert pipeline.prepare_locations(cfg) == 2

    def test_zero_cities_fails(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            pipeline.cities_module,
            "main",
            lambda: Path("cities.csv").write_text("location_id,city,state,lat,lng\n"),
        )
        monkeypatch.setattr(pipeline.locations_module, "main", MagicMock())

        with pytest.raises(pipeline.PipelineError, match="zero locations"):
            pipeline.prepare_locations(_config(["--years", "2024"]))


class TestOutputsAvailable:
    def test_missing_product_output_fails(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)

        with pytest.raises(pipeline.PipelineError, match="No pet batch"):
            pipeline.assert_outputs_available(_config(["--years", "2024"]))

    def test_present_output_passes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        for product in ("pet", "wetbulb"):
            shard_dir = tmp_path / f"{product}_data_csv/year=2024"
            shard_dir.mkdir(parents=True)
            (shard_dir / f"{product}_batch_0000_00.parquet").touch()

        pipeline.assert_outputs_available(_config(["--years", "2024"]))


class TestEra5Pull:
    def test_cloud_run_executes_one_tasked_run_per_year(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        commands: list[list[str]] = []
        monkeypatch.setattr(
            pipeline,
            "_run_command",
            lambda command, **_kwargs: commands.append(command),
        )
        monkeypatch.setattr(pipeline, "sync_outputs_from_s3", MagicMock())

        cfg = _config(["--years", "2023", "2024", "--skip-remote-clear"])
        pipeline.run_era5_pull(cfg)

        gcloud_commands = [c for c in commands if c[0] == "gcloud"]
        assert len(gcloud_commands) == 2
        assert "--tasks" in gcloud_commands[0]
        tasks_value = gcloud_commands[0][gcloud_commands[0].index("--tasks") + 1]
        assert tasks_value == str(pipeline.FULL_TIME_SHARD_COUNT)

    def test_local_mode_runs_all_jobs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        commands: list[list[str]] = []
        monkeypatch.setattr(
            pipeline,
            "_run_command",
            lambda command, **_kwargs: commands.append(command),
        )

        cfg = _config(
            [
                "--local",
                "--years",
                "2024",
                "--era5-city-shard-count",
                "1",
                "--era5-time-shard-count",
                "2",
            ]
        )
        cfg.time_shard_indexes = [0, 1]
        pipeline.run_era5_pull(cfg)

        assert len(commands) == 2

    def test_local_failure_is_reported(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)

        def failing_run(_command: list[str], **_kwargs: object) -> None:
            msg = "job failed with exit code 1."
            raise pipeline.PipelineError(msg)

        monkeypatch.setattr(pipeline, "_run_command", failing_run)

        cfg = _config(["--local", "--years", "2024"])
        cfg.time_shard_indexes = [0]
        with pytest.raises(pipeline.PipelineError, match="local ERA5 job"):
            pipeline.run_era5_pull(cfg)


class TestNldasPull:
    def test_cloud_run_executes_one_tasked_run_per_year(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        commands: list[list[str]] = []
        monkeypatch.setattr(
            pipeline,
            "_run_command",
            lambda command, **_kwargs: commands.append(command),
        )

        cfg = _config(["--years", "2023", "2024", "--skip-remote-clear"])
        pipeline.run_nldas_pull(cfg)

        gcloud_commands = [c for c in commands if c[0] == "gcloud"]
        assert len(gcloud_commands) == 2
        assert all("nldas-worker" in c for c in gcloud_commands)
        tasks_value = gcloud_commands[0][gcloud_commands[0].index("--tasks") + 1]
        assert tasks_value == str(pipeline.FULL_TIME_SHARD_COUNT)

    def test_local_mode_runs_all_jobs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        commands: list[list[str]] = []
        monkeypatch.setattr(
            pipeline,
            "_run_command",
            lambda command, **_kwargs: commands.append(command),
        )

        cfg = _config(
            [
                "--local",
                "--years",
                "2024",
                "--era5-city-shard-count",
                "1",
                "--nldas-time-shard-count",
                "2",
            ]
        )
        cfg.nldas_time_shard_indexes = [0, 1]
        pipeline.run_nldas_pull(cfg)

        assert len(commands) == 2
        assert all("nldas.py" in command[1] for command in commands)

    def test_local_failure_is_reported(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)

        def failing_run(_command: list[str], **_kwargs: object) -> None:
            msg = "job failed with exit code 1."
            raise pipeline.PipelineError(msg)

        monkeypatch.setattr(pipeline, "_run_command", failing_run)

        cfg = _config(["--local", "--years", "2024"])
        cfg.nldas_time_shard_indexes = [0]
        with pytest.raises(pipeline.PipelineError, match="local NLDAS job"):
            pipeline.run_nldas_pull(cfg)


class TestSyncOutputs:
    def test_syncs_each_selected_product(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        commands: list[list[str]] = []
        monkeypatch.setattr(
            pipeline,
            "_run_command",
            lambda command, **_kwargs: commands.append(command),
        )

        pipeline.sync_outputs_from_s3(_config(["--years", "2024"]))

        assert len(commands) == 2
        assert all(command[:3] == ["aws", "s3", "sync"] for command in commands)


class _HistoryCursor:
    def __init__(self, counts: dict[str, int]) -> None:
        self._counts = counts
        self._result = 0

    def execute(self, statement: str) -> None:
        for table, count in self._counts.items():
            if f"public.{table}" in statement:
                self._result = count

    def fetchone(self) -> tuple[int]:
        return (self._result,)

    def __enter__(self) -> _HistoryCursor:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        _ = (exc_type, exc, tb)


class _HistoryConnection:
    def __init__(self, counts: dict[str, int]) -> None:
        self._counts = counts

    def cursor(self) -> _HistoryCursor:
        return _HistoryCursor(self._counts)


class TestHistoryDepth:
    def test_insufficient_pet_history_fails(self) -> None:
        conn = _HistoryConnection({"pet": 3, "wetbulb": 12})

        with pytest.raises(pipeline.PipelineError, match="at least 10 PET years"):
            pipeline._check_history_depth(
                cast("Any", conn), _config(["--years", "2024"])
            )

    def test_short_wetbulb_history_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        conn = _HistoryConnection({"pet": 12, "wetbulb": 2})

        with caplog.at_level(logging.WARNING):
            pipeline._check_history_depth(
                cast("Any", conn), _config(["--years", "2024"])
            )

        assert "wetbulb year(s)" in caplog.text


class TestRunDbLoad:
    def _patch_db(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, MagicMock]:
        mocks = {
            "execute_sql_file": MagicMock(),
            "bulk_insert_csv_files": MagicMock(),
            "_load_table_files": MagicMock(),
            "refresh_materialized_views": MagicMock(),
            "refresh_query_planner_statistics": MagicMock(),
            "_check_history_depth": MagicMock(),
        }
        for name, mock in mocks.items():
            monkeypatch.setattr(pipeline, name, mock)
        monkeypatch.setattr(pipeline, "resolve_database_uri", lambda: "postgresql://x")
        mocks["connect"] = MagicMock()
        monkeypatch.setattr(pipeline.psycopg, "connect", lambda _uri: mocks["connect"])
        return mocks

    def test_recreate_mode_rebuilds_views(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        mocks = self._patch_db(monkeypatch)

        pipeline.run_db_load(_config(["--years", "2024"]))

        executed_files = [
            call.args[1] for call in mocks["execute_sql_file"].call_args_list
        ]
        assert executed_files == [
            "drop_views.sql",
            "create_tables.sql",
            "create_views.sql",
        ]
        assert not mocks["refresh_materialized_views"].called
        assert mocks["connect"].close.called

    def test_refresh_mode_keeps_views(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        mocks = self._patch_db(monkeypatch)

        pipeline.run_db_load(_config(["--years", "2024", "--views", "refresh"]))

        executed_files = [
            call.args[1] for call in mocks["execute_sql_file"].call_args_list
        ]
        assert executed_files == ["create_tables.sql"]
        assert mocks["refresh_materialized_views"].called

    def test_skip_mode_leaves_views_alone(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        mocks = self._patch_db(monkeypatch)

        pipeline.run_db_load(_config(["--years", "2024", "--views", "skip"]))

        executed_files = [
            call.args[1] for call in mocks["execute_sql_file"].call_args_list
        ]
        assert executed_files == ["create_tables.sql"]
        assert not mocks["refresh_materialized_views"].called
        assert not mocks["_check_history_depth"].called

    def test_skips_without_database_credentials(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(pipeline, "resolve_database_uri", lambda: None)
        connect = MagicMock()
        monkeypatch.setattr(pipeline.psycopg, "connect", connect)

        with caplog.at_level(logging.WARNING):
            pipeline.run_db_load(_config(["--years", "2024"]))

        assert not connect.called


class TestRunPipeline:
    def test_stage_order_and_skips(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[str] = []
        monkeypatch.setattr(
            pipeline, "prepare_locations", lambda _cfg: calls.append("locations") or 1
        )
        monkeypatch.setattr(
            pipeline, "run_era5_pull", lambda _cfg: calls.append("pull")
        )
        monkeypatch.setattr(
            pipeline, "run_nldas_pull", lambda _cfg: calls.append("nldas_pull")
        )
        monkeypatch.setattr(
            pipeline, "sync_outputs_from_s3", lambda _cfg: calls.append("sync")
        )
        monkeypatch.setattr(
            pipeline,
            "assert_outputs_available",
            lambda _cfg: calls.append("assert"),
        )
        monkeypatch.setattr(pipeline, "run_db_load", lambda _cfg: calls.append("db"))

        pipeline.run_pipeline(_config(["--years", "2024"]))
        assert calls == ["locations", "pull", "nldas_pull", "sync", "assert", "db"]

        calls.clear()
        pipeline.run_pipeline(_config(["--years", "2024", "--products", "pet"]))
        assert calls == ["locations", "pull", "sync", "assert", "db"]

        calls.clear()
        pipeline.run_pipeline(_config(["--years", "2024", "--products", "wetbulb"]))
        assert calls == ["locations", "nldas_pull", "sync", "assert", "db"]

        calls.clear()
        pipeline.run_pipeline(
            _config(["--years", "2024", "--skip-era5-pull", "--sync-from-s3"])
        )
        assert calls == ["locations", "sync", "assert", "db"]

        calls.clear()
        pipeline.run_pipeline(
            _config(["--years", "2024", "--skip-era5-pull", "--skip-db-load"])
        )
        assert calls == ["locations", "assert"]
