from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from future.config import Settings


def test_defaults() -> None:
    settings = Settings()
    assert settings.seed == 42
    assert settings.start == date(2020, 1, 2)
    assert settings.end == date(2025, 12, 31)
    assert settings.data_source == "mock"


def test_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FUTURE_SEED", "7")
    monkeypatch.setenv("FUTURE_START", "2024-01-01")
    monkeypatch.setenv("FUTURE_END", "2024-06-30")
    monkeypatch.setenv("FUTURE_DATA_SOURCE", "CSV")
    monkeypatch.setenv("FUTURE_DATA_DIR", "tmp/data")

    settings = Settings.from_env()

    assert settings.seed == 7
    assert settings.start == date(2024, 1, 1)
    assert settings.end == date(2024, 6, 30)
    assert settings.data_source == "csv"
    assert settings.data_dir == Path("tmp/data")


def test_invalid_date_range() -> None:
    with pytest.raises(ValueError, match="start"):
        Settings(start=date(2024, 6, 30), end=date(2024, 1, 1))


def test_invalid_source() -> None:
    with pytest.raises(ValueError, match="data_source"):
        Settings(data_source="unknown")


def test_invalid_env_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FUTURE_SEED", "abc")
    with pytest.raises(ValueError, match="FUTURE_SEED"):
        Settings.from_env()
