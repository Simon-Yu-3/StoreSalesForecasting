"""Identify the feature implementation used by a newly generated model cache."""

from hashlib import sha256
from pathlib import Path

FEATURE_SCHEMA_VERSION = 1
SOURCE_DIRECTORY = Path(__file__).parent


def feature_provenance(*source_files: str) -> dict:
    return {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "source_file_sha256": {
            f"auxiliary/{name}": sha256(
                (SOURCE_DIRECTORY / name).read_bytes()
            ).hexdigest()
            for name in source_files
        },
    }
