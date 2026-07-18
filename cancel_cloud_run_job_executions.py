"""Cancel running Cloud Run job executions."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--project")
    parser.add_argument("--gcloud-bin", default="gcloud")
    return parser.parse_args()


def _project_args(project: str | None) -> list[str]:
    return ["--project", project] if project else []


def _list_executions(
    *,
    gcloud_bin: str,
    job: str,
    region: str,
    project: str | None,
) -> list[dict[str, Any]]:
    command = [
        gcloud_bin,
        "run",
        "jobs",
        "executions",
        "list",
        "--job",
        job,
        "--region",
        region,
        *_project_args(project),
        "--format=json",
        "--quiet",
    ]
    result = subprocess.run(  # NOSONAR
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        if "NOT_FOUND" in result.stderr:
            logger.info(
                "Cloud Run job %r not found in %s; nothing to cancel.",
                job,
                region,
            )
            return []
        raise subprocess.CalledProcessError(
            result.returncode,
            command,
            output=result.stdout,
            stderr=result.stderr,
        )

    output = result.stdout.strip()
    if not output:
        return []
    parsed = json.loads(output)
    if not isinstance(parsed, list):
        msg = "Expected Cloud Run executions list output to be a JSON list."
        raise TypeError(msg)
    return parsed


def _is_execution_running(execution: dict[str, Any]) -> bool:
    status = execution.get("status")
    if not isinstance(status, dict):
        return True

    if status.get("completionTime"):
        return False

    conditions = status.get("conditions")
    if not isinstance(conditions, list):
        return True

    for condition in conditions:
        if not isinstance(condition, dict):
            continue
        if condition.get("type") != "Completed":
            continue
        if str(condition.get("status", "")).lower() == "true":
            return False

    return True


def _running_execution_names(executions: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for execution in executions:
        metadata = execution.get("metadata")
        if not isinstance(metadata, dict):
            continue
        name = metadata.get("name")
        if isinstance(name, str) and name and _is_execution_running(execution):
            names.append(name)
    return sorted(names)


def _cancel_execution(
    *,
    gcloud_bin: str,
    execution_name: str,
    region: str,
    project: str | None,
) -> None:
    command = [
        gcloud_bin,
        "run",
        "jobs",
        "executions",
        "cancel",
        execution_name,
        "--region",
        region,
        *_project_args(project),
        "--quiet",
    ]
    subprocess.run(command, check=True)  # NOSONAR


def cancel_running_executions(
    *,
    gcloud_bin: str,
    job: str,
    region: str,
    project: str | None,
) -> list[str]:
    """Cancel running Cloud Run job executions."""
    executions = _list_executions(
        gcloud_bin=gcloud_bin,
        job=job,
        region=region,
        project=project,
    )
    running_execution_names = _running_execution_names(executions)

    if not running_execution_names:
        logger.info(
            "No running Cloud Run executions found for job %r in %s.",
            job,
            region,
        )
        return []

    logger.info(
        "Cancelling %d running Cloud Run execution(s) for job %r in %s...",
        len(running_execution_names),
        job,
        region,
    )
    for execution_name in running_execution_names:
        logger.info(" - cancelling %s", execution_name)
        _cancel_execution(
            gcloud_bin=gcloud_bin,
            execution_name=execution_name,
            region=region,
            project=project,
        )

    return running_execution_names


def main() -> None:
    """Execute the main entry point for cancelling running executions."""
    args = _parse_args()
    cancel_running_executions(
        gcloud_bin=args.gcloud_bin,
        job=args.job,
        region=args.region,
        project=args.project,
    )


if __name__ == "__main__":
    main()
