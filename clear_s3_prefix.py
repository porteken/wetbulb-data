"""Delete S3 objects under a prefix in batches."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

DELETE_BATCH_SIZE = 1000
DEFAULT_MAX_WORKERS = 8
logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger(__name__)


# noinspection PyDeprecation
def _aws_executable() -> str:
    aws_path = shutil.which("aws")
    if aws_path is None:
        msg = "aws CLI is required but was not found on PATH."
        raise RuntimeError(msg)
    return aws_path


def _run_aws_json(args: list[str]) -> dict[str, Any]:
    completed = subprocess.run(  # NOSONAR
        [_aws_executable(), *args, "--output", "json"],
        check=True,
        capture_output=True,
        text=True,
    )
    if not completed.stdout.strip():
        return {}
    return json.loads(completed.stdout)


def _delete_batch(bucket: str, objects: list[dict[str, str]]) -> int:
    payload = {"Objects": objects, "Quiet": True}
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        delete=False,
    ) as handle:
        json.dump(payload, handle)
        handle.flush()
        payload_path = handle.name

    try:
        completed = subprocess.run(
            [
                _aws_executable(),
                "s3api",
                "delete-objects",
                "--bucket",
                bucket,
                "--delete",
                f"file://{payload_path}",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    finally:
        Path(payload_path).unlink()

    response = _load_json_output(completed.stdout)
    failed_objects = _failed_batch_objects(objects, response)
    if completed.returncode == 0 and not failed_objects:
        return len(objects)

    if len(objects) == 1:
        return _delete_single_object(bucket, objects[0], completed, response)

    if failed_objects:
        successful_count = len(objects) - len(failed_objects)
        return successful_count + _delete_in_subbatches(bucket, failed_objects)

    return _delete_in_subbatches(bucket, objects)


def _delete_in_subbatches(bucket: str, objects: list[dict[str, str]]) -> int:
    midpoint = len(objects) // 2
    return _delete_batch(bucket, objects[:midpoint]) + _delete_batch(
        bucket,
        objects[midpoint:],
    )


def _delete_single_object(
    bucket: str,
    object_info: dict[str, str],
    batch_result: subprocess.CompletedProcess[str],
    batch_response: dict[str, Any],
) -> int:
    args = [
        _aws_executable(),
        "s3api",
        "delete-object",
        "--bucket",
        bucket,
        "--key",
        object_info["Key"],
    ]
    version_id = object_info.get("VersionId")
    if version_id is not None:
        args.extend(["--version-id", version_id])

    completed = subprocess.run(
        args,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode == 0:
        return 1

    error_details = _format_delete_error(
        object_info,
        completed.stdout,
        completed.stderr,
    )
    if not error_details:
        error_details = _format_delete_error(
            object_info,
            batch_result.stdout,
            batch_result.stderr,
            response=batch_response,
        )
    raise RuntimeError(error_details)


def _load_json_output(raw_output: str) -> dict[str, Any]:
    if not raw_output.strip():
        return {}
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        return {}


def _failed_batch_objects(
    requested_objects: list[dict[str, str]],
    response: dict[str, Any],
) -> list[dict[str, str]]:
    error_entries = response.get("Errors", [])
    if not error_entries:
        return []

    failed_lookup = {
        (entry["Key"], entry.get("VersionId"))
        for entry in error_entries
        if "Key" in entry
    }
    return [
        object_info
        for object_info in requested_objects
        if (object_info["Key"], object_info.get("VersionId")) in failed_lookup
    ]


def _format_delete_error(
    object_info: dict[str, str],
    stdout: str,
    stderr: str,
    *,
    response: dict[str, Any] | None = None,
) -> str:
    response_payload = response if response is not None else _load_json_output(stdout)
    error_entries = response_payload.get("Errors", [])
    if error_entries:
        first_error = error_entries[0]
        return (
            "Failed to delete "
            f"s3://{object_info['Key']}: "
            f"{first_error.get('Code', 'Unknown')} "
            f"{first_error.get('Message', '').strip()}"
        ).strip()

    for text in (stderr.strip(), stdout.strip()):
        if text:
            return f"Failed to delete s3://{object_info['Key']}: {text}"

    return ""


def _chunked(items: list[dict[str, str]], size: int) -> list[list[dict[str, str]]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _list_versioned_objects(bucket: str, prefix: str) -> list[dict[str, str]]:
    objects: list[dict[str, str]] = []
    key_marker: str | None = None
    version_id_marker: str | None = None

    while True:
        args = [
            "s3api",
            "list-object-versions",
            "--bucket",
            bucket,
            "--prefix",
            prefix,
        ]
        if key_marker is not None:
            args.extend(["--key-marker", key_marker])
        if version_id_marker is not None:
            args.extend(["--version-id-marker", version_id_marker])

        response = _run_aws_json(args)
        objects.extend(
            {
                "Key": entry["Key"],
                "VersionId": entry["VersionId"],
            }
            for entry in response.get("Versions", [])
        )
        objects.extend(
            {
                "Key": entry["Key"],
                "VersionId": entry["VersionId"],
            }
            for entry in response.get("DeleteMarkers", [])
        )

        if not response.get("IsTruncated"):
            return objects

        key_marker = response.get("NextKeyMarker")
        version_id_marker = response.get("NextVersionIdMarker")


def _list_current_objects(bucket: str, prefix: str) -> list[dict[str, str]]:
    objects: list[dict[str, str]] = []
    continuation_token: str | None = None

    while True:
        args = [
            "s3api",
            "list-objects-v2",
            "--bucket",
            bucket,
            "--prefix",
            prefix,
        ]
        if continuation_token is not None:
            args.extend(["--continuation-token", continuation_token])

        response = _run_aws_json(args)
        objects.extend({"Key": entry["Key"]} for entry in response.get("Contents", []))

        if not response.get("IsTruncated"):
            return objects

        continuation_token = response.get("NextContinuationToken")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete all S3 objects under a prefix using batched delete-objects calls.",
    )
    parser.add_argument("--bucket", required=True, help="Bucket name.")
    parser.add_argument("--prefix", required=True, help="Prefix to clear.")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help="Maximum concurrent delete-objects calls.",
    )
    parser.add_argument(
        "--include-versions",
        action="store_true",
        help=(
            "Delete all object versions and delete markers under the prefix. "
            "By default, only current objects are deleted, matching "
            "`aws s3 rm --recursive` behavior on a versioned bucket."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Delete all objects under the S3 prefix."""
    args = _parse_args()
    if args.max_workers < 1:
        msg = "--max-workers must be >= 1"
        raise SystemExit(msg)

    objects = (
        _list_versioned_objects(args.bucket, args.prefix)
        if args.include_versions
        else _list_current_objects(args.bucket, args.prefix)
    )
    if not objects:
        LOGGER.info("No S3 objects found under s3://%s/%s", args.bucket, args.prefix)
        return

    deleted_count = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(_delete_batch, args.bucket, batch)
            for batch in _chunked(objects, DELETE_BATCH_SIZE)
        ]
        for future in futures:
            deleted_count += future.result()

    LOGGER.info(
        "Deleted %s S3 entries under s3://%s/%s",
        deleted_count,
        args.bucket,
        args.prefix,
    )


if __name__ == "__main__":
    main()
