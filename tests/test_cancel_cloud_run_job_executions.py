"""Tests for Cloud Run job cancellation."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from cancel_cloud_run_job_executions import (
    _is_execution_running,
    _list_executions,
    _running_execution_names,
)


class TestIsExecutionRunning:
    def test_running_execution(self) -> None:
        execution = {
            "status": {"conditions": [{"type": "Completed", "status": "False"}]}
        }
        assert _is_execution_running(execution) is True

    def test_completed_execution(self) -> None:
        execution = {"status": {"completionTime": "2024-01-01T00:00:00Z"}}
        assert _is_execution_running(execution) is False

    def test_completed_condition_true(self) -> None:
        execution = {
            "status": {"conditions": [{"type": "Completed", "status": "True"}]}
        }
        assert _is_execution_running(execution) is False

    def test_missing_status(self) -> None:
        assert _is_execution_running({}) is True


class TestRunningExecutionNames:
    def test_filters_running(self) -> None:
        executions = [
            {
                "metadata": {"name": "exec-1"},
                "status": {"completionTime": "2024-01-01T00:00:00Z"},
            },
            {
                "metadata": {"name": "exec-2"},
                "status": {"conditions": [{"type": "Completed", "status": "False"}]},
            },
        ]
        assert _running_execution_names(executions) == ["exec-2"]

    def test_returns_sorted(self) -> None:
        executions = [
            {"metadata": {"name": "b"}, "status": {}},
            {"metadata": {"name": "a"}, "status": {}},
        ]
        assert _running_execution_names(executions) == ["a", "b"]


class TestListExecutions:
    @patch("subprocess.run")
    def test_missing_job_returns_empty_list(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr=(
                "ERROR: (gcloud.run.jobs.executions.list) NOT_FOUND: "
                "Resource 'nldas-worker' of kind 'JOB' in region "
                "'us-east1' in project 'x' does not exist."
            ),
        )

        result = _list_executions(
            gcloud_bin="gcloud",
            job="nldas-worker",
            region="us-east1",
            project="x",
        )

        assert result == []

    @patch("subprocess.run")
    def test_other_failure_raises(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="ERROR: permission denied",
        )

        with pytest.raises(subprocess.CalledProcessError):
            _list_executions(
                gcloud_bin="gcloud",
                job="era5-worker",
                region="us-east1",
                project="x",
            )


@patch("subprocess.run")
def test_cancel_running_executions_integration(mock_run: MagicMock) -> None:
    from cancel_cloud_run_job_executions import cancel_running_executions

    mock_run.return_value.returncode = 0
    mock_run.return_value.stdout = '[{"metadata": {"name": "era5-worker-running"}, "status": {}}, {"metadata": {"name": "era5-worker-running-2"}, "status": {}}]'

    cancelled = cancel_running_executions(
        gcloud_bin="gcloud",
        job="era5-worker",
        region="us-central1",
        project="my-project",
    )

    assert cancelled == [
        "era5-worker-running",
        "era5-worker-running-2",
    ]
