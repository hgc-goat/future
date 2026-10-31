from __future__ import annotations

import json
from pathlib import Path

from future.cli import main
from future.data.storage import write_dataset


def _prepare_data(tmp_path: Path, dataset) -> Path:
    data_dir = tmp_path / "data"
    write_dataset(dataset, data_dir)
    return data_dir


def test_compute_factors_and_backtest(tmp_path: Path, capsys, stage2_dataset) -> None:
    data_dir = _prepare_data(tmp_path, stage2_dataset)
    factors_dir = tmp_path / "factors"
    backtest_dir = tmp_path / "backtest"

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
            "--json",
        ]
    )
    assert code == 0
    generated = json.loads(capsys.readouterr().out)
    assert generated["ok"] is True
    assert generated["rows"] > 0
    assert (factors_dir / "factors.csv").is_file()

    code = main(
        [
            "backtest",
            "--data-dir",
            str(data_dir),
            "--factors",
            str(factors_dir / "factors.csv"),
            "--out",
            str(backtest_dir),
            "--oos-start",
            "2024-01-01",
            "--oos-end",
            "2025-12-31",
            "--json",
        ]
    )
    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["ok"] is True
    assert summary["long_short"]["n_days"] > 0
    assert "20" in summary["oos_ic"]["composite"]
    assert (backtest_dir / "metrics.json").is_file()
    assert (backtest_dir / "layer_returns.csv").is_file()
    assert (backtest_dir / "ic_series.csv").is_file()


def test_backtest_output_reproducible(tmp_path: Path, capsys, stage2_dataset) -> None:
    data_dir = _prepare_data(tmp_path, stage2_dataset)
    factors_dir = tmp_path / "factors"
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

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
    capsys.readouterr()

    for output in (first_dir, second_dir):
        code = main(
            [
                "backtest",
                "--data-dir",
                str(data_dir),
                "--factors",
                str(factors_dir / "factors.csv"),
                "--out",
                str(output),
                "--oos-start",
                "2024-01-01",
                "--oos-end",
                "2025-12-31",
            ]
        )
        assert code == 0
        capsys.readouterr()

    for filename in ("metrics.json", "layer_returns.csv", "ic_series.csv", "manifest.json"):
        assert (first_dir / filename).read_bytes() == (second_dir / filename).read_bytes()


def test_backtest_missing_factors(tmp_path: Path, capsys) -> None:
    code = main(
        [
            "backtest",
            "--data-dir",
            str(tmp_path),
            "--factors",
            str(tmp_path / "missing.csv"),
            "--out",
            str(tmp_path / "out"),
        ]
    )
    assert code == 1
    assert "回测失败" in capsys.readouterr().err


def test_invalid_layer_count(tmp_path: Path, capsys, stage2_dataset) -> None:
    data_dir = _prepare_data(tmp_path, stage2_dataset)
    factors_dir = tmp_path / "factors"
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
    capsys.readouterr()

    code = main(
        [
            "backtest",
            "--data-dir",
            str(data_dir),
            "--factors",
            str(factors_dir / "factors.csv"),
            "--out",
            str(tmp_path / "out"),
            "--layers",
            "1",
        ]
    )
    assert code == 1
    assert "回测失败" in capsys.readouterr().err
