from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from future.data.base import Dataset, DataSourceError
from future.data.mock import MockDataAdapter
from future.data.registry import resolve_adapter
from future.data.storage import write_dataset
from future.data.validate import validate_dataset


def test_mock_resolution() -> None:
    resolution = resolve_adapter(
        "mock",
        seed=7,
        start=date(2024, 1, 1),
        end=date(2024, 3, 31),
    )

    assert isinstance(resolution.adapter, MockDataAdapter)
    assert not resolution.fallback_planned
    dataset = resolution.adapter.load()
    assert validate_dataset(dataset).ok


def test_real_source_falls_back_to_mock() -> None:
    resolution = resolve_adapter(
        "real",
        seed=7,
        start=date(2024, 1, 1),
        end=date(2024, 3, 31),
    )

    assert resolution.fallback_planned
    dataset = resolution.adapter.load()

    assert validate_dataset(dataset).ok
    assert dataset.metadata["fallback_from"] == "real"
    assert "尚未配置" in dataset.metadata["fallback_reason"]
    assert getattr(resolution.adapter, "fallback_reason", None)


def test_csv_resolution(tmp_path: Path, small_dataset: Dataset) -> None:
    write_dataset(small_dataset, tmp_path)
    resolution = resolve_adapter("csv", data_dir=tmp_path)

    assert resolution.adapter.name == "csv"
    dataset = resolution.adapter.load()
    assert dataset.metadata["source"] == "csv"
    assert validate_dataset(dataset).ok


def test_unknown_source() -> None:
    with pytest.raises(ValueError, match="不支持"):
        resolve_adapter("unknown")


def test_missing_csv_directory(tmp_path: Path) -> None:
    resolution = resolve_adapter("csv", data_dir=tmp_path)
    with pytest.raises(DataSourceError, match="缺少文件"):
        resolution.adapter.load()
