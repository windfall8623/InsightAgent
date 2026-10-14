"""Shared fixtures.

The sandbox database is built once per test session so the whole suite (and the
51-case evaluation) runs against identical data — a prerequisite for the
reproducibility claim.

The database is created under ``.testdata/`` rather than pytest's ``tmp_path``:
this environment denies access to the shared temp root, and a session-scoped
fixture needs a directory that outlives a single test anyway.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from insight_agent.data.database import connect, ensure_database

TEST_DATA_DIR = Path(__file__).resolve().parents[1] / ".testdata"


@pytest.fixture(scope="session")
def db_path() -> Path:
    """Build a dedicated database once and reuse it for the whole session."""

    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return ensure_database(TEST_DATA_DIR / "test.db")


@pytest.fixture(scope="session")
def conn(db_path: Path) -> Iterator[sqlite3.Connection]:
    c = connect(db_path)
    yield c
    c.close()


@pytest.fixture()
def store_dir() -> Iterator[Path]:
    """A per-test directory for memory files, so tests cannot leak state."""

    path = TEST_DATA_DIR / "memory"
    path.mkdir(parents=True, exist_ok=True)
    yield path
    for leftover in path.glob("*.json"):
        leftover.unlink()


@pytest.fixture(autouse=True)
def _clean_test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure every test runs offline without reading local .env or calling external endpoints."""
    import insight_agent.config as config_module

    monkeypatch.setattr(config_module, "DEFAULT_DOTENV", Path("nonexistent-for-tests/.env"))
    for var in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "INSIGHT_API_KEY",
        "API_KEY",
        "INSIGHT_PROVIDER",
        "INSIGHT_MODEL",
        "INSIGHT_BASE_URL",
        "INSIGHT_ANTHROPIC_BASE_URL",
        "INSIGHT_OPENAI_BASE_URL",
        "ANTHROPIC_BASE_URL",
        "OPENAI_BASE_URL",
    ):
        monkeypatch.delenv(var, raising=False)

