# Keep in sync with requires-python in pyproject.toml: shards.py uses PEP 758
# syntax (unparenthesized except) that only parses on Python 3.14+.
FROM python:3.14-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.9.9 /uv /uvx /bin/

# build-essential + python3-dev: numpy/numcodecs have no prebuilt wheels yet
# for this Python version, so they must build from their hash-pinned sdists.
# libhdf5-dev: h5py's netCDF4 backend falls back to a source build (needing
# the HDF5 headers) whenever no prebuilt wheel exists yet for this Python
# version, same situation as numcodecs above.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libhdf5-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY uv.lock ./

RUN python -m venv "$VIRTUAL_ENV"

RUN cat <<'PY' > /tmp/prepare-requirements.py
from pathlib import Path

WORKER_PACKAGES = (
    "dask",
    "gcsfs",
    "h5netcdf",
    "h5py",
    "numpy",
    "pandas",
    "pyarrow",
    "requests",
    "thermofeel",
    "tqdm",
    "xarray",
    "zarr",
)

lock_content = Path("uv.lock").read_text(encoding="utf-8")
versions = {}
current_name = None
for raw_line in lock_content.splitlines():
    line = raw_line.strip()
    if line.startswith('name = "'):
        current_name = line.split('"', 2)[1]
        continue
    if current_name and line.startswith('version = "'):
        versions[current_name] = line.split('"', 2)[1]
        current_name = None

missing = [name for name in WORKER_PACKAGES if name not in versions]
if missing:
    raise SystemExit(f"Missing Cloud Run worker packages in uv.lock: {missing}")

Path("/tmp/reqs.in").write_text("\n".join(WORKER_PACKAGES))
Path("/tmp/constraints.txt").write_text("\n".join(f"{pkg}=={ver}" for pkg, ver in versions.items()))
print(versions.get("asciitree", "0.3.3"))
PY

RUN python /tmp/prepare-requirements.py > /tmp/asciitree_ver.txt && \
    uv pip compile --generate-hashes /tmp/reqs.in \
        --constraint /tmp/constraints.txt \
        -o /tmp/cloudrun-worker-requirements.txt && \
    uv pip install --python "$VIRTUAL_ENV/bin/python" "asciitree==$(cat /tmp/asciitree_ver.txt)"

# --no-build is intentionally omitted: numcodecs (a transitive zarr
# dependency) has no prebuilt wheel for this Python version yet, only a
# hash-pinned sdist, so it must be built here (gcc is installed above for
# exactly this). --require-hashes still pins every package to uv.lock.
# Dockerfiles don't support trailing/NOSONAR comments, so this exception is
# tracked and accepted directly in SonarQube rather than suppressed inline.
RUN uv pip install --require-hashes \
    --python "$VIRTUAL_ENV/bin/python" \
    --requirement /tmp/cloudrun-worker-requirements.txt

RUN "$VIRTUAL_ENV/bin/python" - <<'PY'
import dask
import gcsfs
import h5netcdf
import h5py
import numpy
import pandas
import pyarrow
import requests
import thermofeel
import tqdm
import xarray
import zarr

print("Cloud Run worker dependencies verified.")
PY

FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY entrypoint.sh cities.py google_era5.py nldas.py partition_io.py pet_corrected.py shards.py wetbulb.py ./

RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && chown -R appuser:appuser /app \
    && chmod +x entrypoint.sh

USER appuser

ENTRYPOINT ["./entrypoint.sh"]