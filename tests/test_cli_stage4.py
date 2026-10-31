from __future__ import annotations

import json
from pathlib import Path

import pytest

from future.cli import main
from future.data.storage import write_dataset

pytest.importorskip("sklearn")


def _prepare(tmp_path: Path, dataset, capsys) -> tuple[Path, Path]:
    data_dir = tmp_path / "data"
    factors_dir = tmp_path / "factors"
    fused_dir = tmp_path / "fused"
    analysis_dir = tmp_path / "news_analysis"
    write_dataset(dataset, data_dir)

    assert (
        main(
            [
                "compute-factors",
                "--data-dir",
                str(data_dir),
                "--out",
                str(factors_dir),
                "--start",
                "2023-01-01",
                "--end",
                "2025-12-31",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert (
        main(
            [
                "fuse-news",
                "--data-dir",
                str(data_dir),
                "--factors",
                str(factors_dir / "factors.csv"),
                "--out",
                str(fused_dir),
                "--analysis-out",
                str(analysis_dir),
                "--oos-start",
                "2024-01-01",
                "--oos-end",
                "2025-12-31",
            ]
        )
        == 0
    )
    capsys.readouterr()
    return data_dir, fused_dir / "fused_factors.csv"


def test_advanced_model_cli(tmp_path: Path, capsys, stage2_dataset) -> None:
    data_dir, fused_path = _prepare(tmp_path, stage2_dataset, capsys)
    output = tmp_path / "advanced"

    code = main(
        [
            "advanced-model",
            "--data-dir",
            str(data_dir),
            "--factors",
            str(fused_path),
            "--out",
            str(output),
            "--train-start",
            "2022-01-01",
            "--train-end",
            "2023-12-31",
            "--oos-start",
            "2024-01-01",
            "--oos-end",
            "2025-12-31",
            "--json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["train_rows"] > 0
    assert payload["baseline_ic20"] is not None
    assert payload["hgb_ic20"] is not None
    assert payload["train_label_end"] <= "2023-12-31"
    assert (output / "predictions.csv").is_file()
    assert (output / "metrics.json").is_file()
    assert (output / "manifest.json").is_file()


def test_advanced_model_reproducible(tmp_path: Path, capsys, stage2_dataset) -> None:
    data_dir, fused_path = _prepare(tmp_path, stage2_dataset, capsys)
    first = tmp_path / "first"
    second = tmp_path / "second"

    for output in (first, second):
        code = main(
            [
                "advanced-model",
                "--data-dir",
                str(data_dir),
                "--factors",
                str(fused_path),
                "--out",
                str(output),
                "--train-start",
                "2022-01-01",
                "--train-end",
                "2023-12-31",
                "--oos-start",
                "2024-01-01",
                "--oos-end",
                "2025-12-31",
            ]
        )
        assert code == 0
        capsys.readouterr()

    for filename in ("predictions.csv", "metrics.json", "manifest.json"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()


def test_drift_monitor_cli(tmp_path: Path, capsys, stage2_dataset) -> None:
    data_dir, fused_path = _prepare(tmp_path, stage2_dataset, capsys)
    output = tmp_path / "drift"

    code = main(
        [
            "drift-monitor",
            "--data-dir",
            str(data_dir),
            "--factors",
            str(fused_path),
            "--out",
            str(output),
            "--train-start",
            "2022-01-01",
            "--train-end",
            "2023-12-31",
            "--oos-start",
            "2024-01-01",
            "--oos-end",
            "2025-12-31",
            "--json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert 0.0 <= payload["severity"]["overall"] <= 1.0
    assert payload["severity"]["level"] in {"low", "medium", "high"}
    assert (output / "drift.json").is_file()
    assert (output / "rolling_ic.csv").is_file()
    assert (output / "manifest.json").is_file()


def test_advanced_model_missing_factors(tmp_path: Path, capsys) -> None:
    code = main(
        [
            "advanced-model",
            "--data-dir",
            str(tmp_path),
            "--factors",
            str(tmp_path / "missing.csv"),
            "--out",
            str(tmp_path / "advanced"),
        ]
    )
    assert code == 1
    assert "进阶模型失败" in capsys.readouterr().err


def test_drift_monitor_missing_factors(tmp_path: Path, capsys) -> None:
    code = main(
        [
            "drift-monitor",
            "--data-dir",
            str(tmp_path),
            "--factors",
            str(tmp_path / "missing.csv"),
            "--out",
            str(tmp_path / "drift"),
        ]
    )
    assert code == 1
    assert "漂移监控失败" in capsys.readouterr().err
