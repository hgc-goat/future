from __future__ import annotations

import json
from pathlib import Path

from future.cli import main
from future.data.storage import write_dataset


def _prepare(tmp_path: Path, dataset, capsys) -> tuple[Path, Path]:
    data_dir = tmp_path / "data"
    factors_dir = tmp_path / "factors"
    write_dataset(dataset, data_dir)

    code = main(
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
    assert code == 0
    capsys.readouterr()
    return data_dir, factors_dir


def test_fuse_news(tmp_path: Path, capsys, stage2_dataset) -> None:
    data_dir, factors_dir = _prepare(tmp_path, stage2_dataset, capsys)
    fused_dir = tmp_path / "fused"
    analysis_dir = tmp_path / "analysis"

    code = main(
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
            "--json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["ok"] is True
    assert payload["coverage"]["total_rows"] > 0
    assert payload["base_ic20"] is not None
    assert payload["fused_ic20"] is not None
    assert (fused_dir / "fused_factors.csv").is_file()
    assert (fused_dir / "manifest.json").is_file()
    assert (analysis_dir / "analysis.json").is_file()
    assert (analysis_dir / "manifest.json").is_file()


def test_fuse_news_reproducible(tmp_path: Path, capsys, stage2_dataset) -> None:
    data_dir, factors_dir = _prepare(tmp_path, stage2_dataset, capsys)
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_analysis = tmp_path / "first_analysis"
    second_analysis = tmp_path / "second_analysis"

    for fused_dir, analysis_dir in (
        (first, first_analysis),
        (second, second_analysis),
    ):
        code = main(
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
        assert code == 0
        capsys.readouterr()

    for filename in ("fused_factors.csv", "manifest.json"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()
    for filename in ("analysis.json", "manifest.json"):
        assert (first_analysis / filename).read_bytes() == (second_analysis / filename).read_bytes()


def test_fuse_news_missing_factors(tmp_path: Path, capsys) -> None:
    code = main(
        [
            "fuse-news",
            "--data-dir",
            str(tmp_path),
            "--factors",
            str(tmp_path / "missing.csv"),
            "--out",
            str(tmp_path / "out"),
            "--analysis-out",
            str(tmp_path / "analysis"),
        ]
    )
    assert code == 1
    assert "新闻融合失败" in capsys.readouterr().err


def test_fuse_news_custom_thresholds(tmp_path: Path, capsys, stage2_dataset) -> None:
    data_dir, factors_dir = _prepare(tmp_path, stage2_dataset, capsys)

    code = main(
        [
            "fuse-news",
            "--data-dir",
            str(data_dir),
            "--factors",
            str(factors_dir / "factors.csv"),
            "--out",
            str(tmp_path / "fused"),
            "--analysis-out",
            str(tmp_path / "analysis"),
            "--oos-start",
            "2024-01-01",
            "--oos-end",
            "2025-12-31",
            "--min-news-count",
            "999",
            "--json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["coverage"]["eligible_rows"] == 0
    assert payload["coverage"]["confirmed_rows"] == 0
