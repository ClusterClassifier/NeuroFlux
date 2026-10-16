"""Atomic status and configuration output for ablation runs."""

import json
from pathlib import Path


def _v1_atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(path)
