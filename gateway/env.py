"""Load ``KEY=VALUE`` lines from a ``.env`` file into the process environment.

Deliberately tiny, so the project needs no extra dependency. Rules:

* Lines that are blank or start with ``#`` are ignored.
* Surrounding single or double quotes around the value are stripped.
* A variable already set in the real environment is never overwritten, so a
  shell ``export`` or a CI secret always wins over the file.

The ``.env`` file itself is git-ignored. ``.env.example`` shows the expected names.
"""

from __future__ import annotations

import os
from pathlib import Path

from .config import ROOT

DEFAULT_ENV_PATH = ROOT / ".env"


def parse_dotenv(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def load_dotenv(path: Path | str = DEFAULT_ENV_PATH, *, override: bool = False) -> dict[str, str]:
    """Read ``path`` into ``os.environ``. Returns the variables that were set."""
    path = Path(path)
    if not path.exists():
        return {}
    applied: dict[str, str] = {}
    for key, value in parse_dotenv(path.read_text(encoding="utf-8")).items():
        if override or key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    return applied
