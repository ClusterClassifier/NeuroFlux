"""Load the exact embedded V1 snapshots used by the original ablation runner."""

import json
from pathlib import Path

# Preserves the original source strings so configuration hashes remain identical.
_V1_SOURCE_CELLS = json.loads(
    Path(__file__).with_name("source_cells.json").read_text(encoding="utf-8")
)
