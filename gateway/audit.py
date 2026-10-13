"""The audit log: an append-only file of decision records.

One JSON object per line (the "JSON Lines" format, ``.jsonl``). Appending a line
never rewrites earlier lines, so a crash mid-write cannot corrupt history, and a
reader can stream the file without loading all of it. This is the file the
compliance reviewer inspects and the dashboard aggregates.

The cache is a different thing: it stores *answers* for reuse. This stores
*decisions* for accountability, and nothing is ever read back to answer a user.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from .config import ROOT
from .schema import DecisionRecord

DEFAULT_LOG_PATH = ROOT / "var" / "decisions.jsonl"


class AuditLog:
    """Append decision records to a JSON Lines file and read them back."""

    def __init__(self, path: Path | str = DEFAULT_LOG_PATH) -> None:
        self.path = Path(path)

    def append(self, record: DecisionRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = record.to_json(indent=None)  # indent=None keeps the record on one line
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def read(self) -> Iterator[dict]:
        """Yield records oldest first. A missing file simply yields nothing."""
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def __len__(self) -> int:
        return sum(1 for _ in self.read())
