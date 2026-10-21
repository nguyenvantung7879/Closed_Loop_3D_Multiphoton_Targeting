"""Shared report serialization and streaming content hashes.

Scientific fingerprints retain their own schema in the calling module.
"""
from pathlib import Path
import hashlib
import json


def sha256_file(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value, *, default=None):
    """Write finite JSON; keep string coercion an explicit caller decision."""
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=default),
                          encoding='utf-8')
