"""
Central secret loader. Keys come from environment variables, falling back to a
gitignored `.secrets.env` file at the project root. Never hard-code keys here.

Setup:
  cp .secrets.env.example .secrets.env   # then paste your real keys
"""

import os
from pathlib import Path

_SECRETS_FILE = Path(__file__).parent.parent / ".secrets.env"


def _load_secrets_file() -> None:
    """Load KEY=VALUE lines from .secrets.env into the environment (no overwrite)."""
    if not _SECRETS_FILE.exists():
        return
    for line in _SECRETS_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


_load_secrets_file()


def require_key(name: str) -> str:
    """Return a key or raise a clear, actionable error if it is missing."""
    value = os.environ.get(name)
    if not value or value.startswith("your_"):
        raise RuntimeError(
            f"Missing API key '{name}'. Copy .secrets.env.example to .secrets.env "
            f"and set {name}, or export it as an environment variable."
        )
    return value


def get_key(name: str) -> str | None:
    """Return a key or None (for optional integrations)."""
    value = os.environ.get(name)
    return None if (not value or value.startswith("your_")) else value


FOOTBALL_DATA_API_KEY = "FOOTBALL_DATA_API_KEY"
API_FOOTBALL_KEY      = "API_FOOTBALL_KEY"
