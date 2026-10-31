from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from future.data.base import Dataset, DataSourceError
from future.data.storage import (
    FILE_NAMES,
    CsvDataAdapter,
    read_dataset,
    read_manifest,
    verify_manifest,
    write_dataset,
)
from future.data.validate import validate_dataset


def test_write_and_read_roundtrip(tmp_path: Path, small_dataset: Dataset) -> None:
    manifest = write_dataset(small_dataset, tmp_path)

    assert manifest["schema_version"] == 1
    assert manifest["seed"] == 7
    assert (tmp_path / "manifest.json").is_file()
    assert verify_manifest(tmp_path) == []

    loaded = read_dataset(tmp_path)
    report = validate_dataset(loaded)
    assert report.ok, report.errors
    assert len(loaded.prices) == len(small_dataset.prices)
    assert len(loaded.news) == len(small_dataset.news)


def test_files_are_reproducible(tmp_path: Path, small_dataset: Dataset) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_dataset(small_dataset, first)
    write_dataset(small_dataset, second)

    for filename in FILE_NAMES:
        assert (first / filename).read_bytes() == (second / filename).read_bytes()
    assert read_manifest(first) == read_manifest(second)


def test_manifest_detects_tamper(tmp_path: Path, small_dataset: Dataset) -> None:
    write_dataset(small_dataset, tmp_path)
    with (tmp_path / "prices.csv").open("a", encoding="utf-8") as handle:
        handle.write("\n")

    errors = verify_manifest(tmp_path)
    assert any("SHA-256" in error for error in errors)


def test_missing_file(tmp_path: Path, small_dataset: Dataset) -> None:
    write_dataset(small_dataset, tmp_path)
    (tmp_path / "news.csv").unlink()

    with pytest.raises(DataSourceError, match="缺少文件"):
        read_dataset(tmp_path)


def test_csv_adapter_loads_range(tmp_path: Path, small_dataset: Dataset) -> None:
    write_dataset(small_dataset, tmp_path)
    adapter = CsvDataAdapter(tmp_path)
    start = date(2024, 2, 1)
    end = date(2024, 5, 31)

    dataset = adapter.load(start, end)

    assert dataset.metadata["source"] == "csv"
    assert dataset.prices["date"].min().date() >= start
    assert dataset.prices["date"].max().date() <= end
    assert dataset.prices["date"].min().month == 2
    assert dataset.prices["date"].max().month == 5
    assert validate_dataset(dataset).ok


def test_csv_adapter_rejects_invalid_schema(tmp_path: Path, small_dataset: Dataset) -> None:
    write_dataset(small_dataset, tmp_path)
    (tmp_path / "prices.csv").write_text("date,sector_id\n", encoding="utf-8")

    with pytest.raises(DataSourceError, match="校验失败"):
        CsvDataAdapter(tmp_path).load()
