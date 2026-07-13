"""Tests for batched S3 prefix deletion helpers."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

import clear_s3_prefix


def _completed_process(
    *,
    stdout: str = "",
    stderr: str = "",
    returncode: int = 0,
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["aws"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


class TestAwsExecutable:
    def test_returns_aws_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            clear_s3_prefix.shutil, "which", lambda _name: "/usr/bin/aws"
        )

        assert clear_s3_prefix._aws_executable() == "/usr/bin/aws"

    def test_raises_when_aws_is_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(clear_s3_prefix.shutil, "which", lambda _name: None)

        with pytest.raises(RuntimeError, match="aws CLI is required"):
            clear_s3_prefix._aws_executable()


class TestRunAwsJson:
    def test_parses_json_output(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(clear_s3_prefix, "_aws_executable", lambda: "aws")

        def fake_run(
            command: list[str],
            *,
            check: bool,
            capture_output: bool,
            text: bool,
        ) -> subprocess.CompletedProcess[str]:
            assert command == ["aws", "s3api", "list-objects-v2", "--output", "json"]
            assert check is True
            assert capture_output is True
            assert text is True
            return _completed_process(stdout='{"Contents": [{"Key": "one"}]}')

        monkeypatch.setattr(clear_s3_prefix.subprocess, "run", fake_run)

        assert clear_s3_prefix._run_aws_json(["s3api", "list-objects-v2"]) == {
            "Contents": [{"Key": "one"}],
        }

    def test_returns_empty_dict_for_blank_stdout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(clear_s3_prefix, "_aws_executable", lambda: "aws")

        def fake_run(
            command: list[str],
            *,
            check: bool,
            capture_output: bool,
            text: bool,
        ) -> subprocess.CompletedProcess[str]:
            _ = (command, check, capture_output, text)
            return _completed_process(stdout="   ")

        monkeypatch.setattr(
            clear_s3_prefix.subprocess,
            "run",
            fake_run,
        )

        assert clear_s3_prefix._run_aws_json(["s3api", "list-objects-v2"]) == {}


class TestDeleteHelpers:
    def test_load_json_output_handles_empty_and_invalid_json(self) -> None:
        assert clear_s3_prefix._load_json_output("") == {}
        assert clear_s3_prefix._load_json_output("not-json") == {}

    def test_failed_batch_objects_filters_to_error_entries(self) -> None:
        requested = [
            {"Key": "one", "VersionId": "1"},
            {"Key": "two", "VersionId": "2"},
            {"Key": "three"},
        ]
        response = {
            "Errors": [
                {"Key": "two", "VersionId": "2", "Code": "Denied"},
                {"Key": "three", "Code": "Denied"},
            ]
        }

        assert clear_s3_prefix._failed_batch_objects(requested, response) == [
            {"Key": "two", "VersionId": "2"},
            {"Key": "three"},
        ]

    def test_format_delete_error_prefers_structured_response(self) -> None:
        assert (
            clear_s3_prefix._format_delete_error(
                {"Key": "some/key"},
                "",
                "",
                response={
                    "Errors": [
                        {
                            "Key": "some/key",
                            "Code": "AccessDenied",
                            "Message": "Forbidden",
                        },
                    ],
                },
            )
            == "Failed to delete s3://some/key: AccessDenied Forbidden"
        )

    def test_format_delete_error_falls_back_to_stderr_then_stdout(self) -> None:
        assert (
            clear_s3_prefix._format_delete_error(
                {"Key": "some/key"},
                "stdout message",
                "stderr message",
            )
            == "Failed to delete s3://some/key: stderr message"
        )

        assert (
            clear_s3_prefix._format_delete_error(
                {"Key": "some/key"},
                "stdout message",
                "",
            )
            == "Failed to delete s3://some/key: stdout message"
        )

    def test_chunked_splits_into_batches(self) -> None:
        items = [
            {"Key": "one"},
            {"Key": "two"},
            {"Key": "three"},
        ]

        assert clear_s3_prefix._chunked(items, 2) == [
            [{"Key": "one"}, {"Key": "two"}],
            [{"Key": "three"}],
        ]

    def test_delete_batch_writes_payload_and_counts_successes(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(clear_s3_prefix, "_aws_executable", lambda: "aws")
        payloads: list[dict[str, object]] = []

        def fake_run(
            command: list[str],
            *,
            check: bool,
            capture_output: bool,
            text: bool,
        ) -> subprocess.CompletedProcess[str]:
            _ = (check, capture_output, text)
            payload_arg = command[-1]
            payload_path = Path(payload_arg.removeprefix("file://"))
            payloads.append(json.loads(payload_path.read_text(encoding="utf-8")))
            return _completed_process(stdout="{}")

        monkeypatch.setattr(clear_s3_prefix.subprocess, "run", fake_run)

        objects = [{"Key": "one"}, {"Key": "two"}]

        assert clear_s3_prefix._delete_batch("bucket", objects) == 2
        assert payloads == [{"Objects": objects, "Quiet": True}]

    def test_delete_batch_retries_single_object_delete(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(clear_s3_prefix, "_aws_executable", lambda: "aws")
        calls: list[list[str]] = []
        responses = iter(
            [
                _completed_process(returncode=1),
                _completed_process(returncode=0),
            ],
        )

        def fake_run(
            command: list[str],
            *,
            check: bool,
            capture_output: bool,
            text: bool,
        ) -> subprocess.CompletedProcess[str]:
            _ = (check, capture_output, text)
            calls.append(command)
            return next(responses)

        monkeypatch.setattr(clear_s3_prefix.subprocess, "run", fake_run)

        deleted = clear_s3_prefix._delete_batch(
            "bucket",
            [{"Key": "one", "VersionId": "abc"}],
        )

        assert deleted == 1
        assert calls[1] == [
            "aws",
            "s3api",
            "delete-object",
            "--bucket",
            "bucket",
            "--key",
            "one",
            "--version-id",
            "abc",
        ]

    def test_delete_single_object_raises_with_batch_error_details(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(clear_s3_prefix, "_aws_executable", lambda: "aws")

        def fake_run(
            command: list[str],
            *,
            check: bool,
            capture_output: bool,
            text: bool,
        ) -> subprocess.CompletedProcess[str]:
            _ = (command, check, capture_output, text)
            return _completed_process(returncode=1)

        monkeypatch.setattr(
            clear_s3_prefix.subprocess,
            "run",
            fake_run,
        )

        with pytest.raises(RuntimeError, match="AccessDenied Denied"):
            clear_s3_prefix._delete_single_object(
                "bucket",
                {"Key": "one"},
                _completed_process(returncode=1),
                {
                    "Errors": [
                        {"Key": "one", "Code": "AccessDenied", "Message": "Denied"},
                    ],
                },
            )


class TestObjectListing:
    def test_list_versioned_objects_handles_pagination(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[list[str]] = []
        responses = iter(
            [
                {
                    "Versions": [{"Key": "one", "VersionId": "1"}],
                    "DeleteMarkers": [{"Key": "two", "VersionId": "2"}],
                    "IsTruncated": True,
                    "NextKeyMarker": "next-key",
                    "NextVersionIdMarker": "next-version",
                },
                {
                    "Versions": [{"Key": "three", "VersionId": "3"}],
                    "DeleteMarkers": [],
                    "IsTruncated": False,
                },
            ],
        )

        def fake_run_aws_json(args: list[str]) -> dict[str, object]:
            calls.append(args)
            return next(responses)

        monkeypatch.setattr(clear_s3_prefix, "_run_aws_json", fake_run_aws_json)

        assert clear_s3_prefix._list_versioned_objects("bucket", "prefix/") == [
            {"Key": "one", "VersionId": "1"},
            {"Key": "two", "VersionId": "2"},
            {"Key": "three", "VersionId": "3"},
        ]
        assert calls == [
            [
                "s3api",
                "list-object-versions",
                "--bucket",
                "bucket",
                "--prefix",
                "prefix/",
            ],
            [
                "s3api",
                "list-object-versions",
                "--bucket",
                "bucket",
                "--prefix",
                "prefix/",
                "--key-marker",
                "next-key",
                "--version-id-marker",
                "next-version",
            ],
        ]

    def test_list_current_objects_handles_pagination(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[list[str]] = []
        responses = iter(
            [
                {
                    "Contents": [{"Key": "one"}],
                    "IsTruncated": True,
                    "NextContinuationToken": "token-1",
                },
                {
                    "Contents": [{"Key": "two"}],
                    "IsTruncated": False,
                },
            ],
        )

        def fake_run_aws_json(args: list[str]) -> dict[str, object]:
            calls.append(args)
            return next(responses)

        monkeypatch.setattr(clear_s3_prefix, "_run_aws_json", fake_run_aws_json)

        assert clear_s3_prefix._list_current_objects("bucket", "prefix/") == [
            {"Key": "one"},
            {"Key": "two"},
        ]
        assert calls == [
            [
                "s3api",
                "list-objects-v2",
                "--bucket",
                "bucket",
                "--prefix",
                "prefix/",
            ],
            [
                "s3api",
                "list-objects-v2",
                "--bucket",
                "bucket",
                "--prefix",
                "prefix/",
                "--continuation-token",
                "token-1",
            ],
        ]


class _FakeFuture:
    def __init__(self, value: int) -> None:
        self._value = value

    def result(self) -> int:
        return self._value


class _FakeExecutor:
    def __init__(self, *, max_workers: int) -> None:
        self.max_workers = max_workers

    def submit(
        self,
        fn: Callable[..., int],
        *args: object,
        **kwargs: object,
    ) -> _FakeFuture:
        _ = self.max_workers
        return _FakeFuture(fn(*args, **kwargs))

    def __enter__(self) -> _FakeExecutor:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        _ = (exc_type, exc, tb)


class TestMain:
    def test_raises_for_invalid_max_workers(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            clear_s3_prefix,
            "_parse_args",
            lambda: argparse.Namespace(
                bucket="bucket",
                prefix="prefix/",
                max_workers=0,
                include_versions=False,
            ),
        )

        with pytest.raises(SystemExit, match="--max-workers must be >= 1"):
            clear_s3_prefix.main()

    def test_logs_when_no_objects_found(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        messages: list[str] = []
        monkeypatch.setattr(
            clear_s3_prefix,
            "_parse_args",
            lambda: argparse.Namespace(
                bucket="bucket",
                prefix="prefix/",
                max_workers=2,
                include_versions=False,
            ),
        )
        monkeypatch.setattr(clear_s3_prefix, "_list_current_objects", lambda *_args: [])
        monkeypatch.setattr(
            clear_s3_prefix.LOGGER,
            "info",
            lambda message, *args: messages.append(message % args),
        )

        clear_s3_prefix.main()

        assert messages == ["No S3 objects found under s3://bucket/prefix/"]

    def test_deletes_objects_and_logs_total(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        messages: list[str] = []
        monkeypatch.setattr(
            clear_s3_prefix,
            "_parse_args",
            lambda: argparse.Namespace(
                bucket="bucket",
                prefix="prefix/",
                max_workers=3,
                include_versions=True,
            ),
        )
        monkeypatch.setattr(
            clear_s3_prefix,
            "_list_versioned_objects",
            lambda *_args: [{"Key": "one"}, {"Key": "two"}, {"Key": "three"}],
        )
        monkeypatch.setattr(
            clear_s3_prefix,
            "_chunked",
            lambda objects, _size: [objects[:2], objects[2:]],
        )
        monkeypatch.setattr(
            clear_s3_prefix,
            "_delete_batch",
            lambda _bucket, batch: len(batch),
        )
        monkeypatch.setattr(clear_s3_prefix, "ThreadPoolExecutor", _FakeExecutor)
        monkeypatch.setattr(
            clear_s3_prefix.LOGGER,
            "info",
            lambda message, *args: messages.append(message % args),
        )

        clear_s3_prefix.main()

        assert messages == ["Deleted 3 S3 entries under s3://bucket/prefix/"]
