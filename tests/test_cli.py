from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from future.cli import main


def test_generate_and_validate(tmp_path: Path, capsys) -> None:
    code = main(
        [
            "generate-data",
            "--seed",
            "7",
            "--start",
            "2024-01-01",
            "--end",
            "2024-03-31",
            "--out",
            str(tmp_path),
            "--json",
        ]
    )
    assert code == 0
    generated = json.loads(capsys.readouterr().out)
    assert generated["ok"] is True
    assert generated["files"]["prices.csv"] > 0
    assert (tmp_path / "manifest.json").is_file()

    code = main(["validate-data", "--path", str(tmp_path), "--json"])
    assert code == 0
    validated = json.loads(capsys.readouterr().out)
    assert validated["ok"] is True
    assert validated["errors"] == []


def test_validate_corrupt_data(tmp_path: Path, capsys) -> None:
    main(
        [
            "generate-data",
            "--seed",
            "7",
            "--start",
            "2024-01-01",
            "--end",
            "2024-03-31",
            "--out",
            str(tmp_path),
        ]
    )
    capsys.readouterr()
    (tmp_path / "prices.csv").write_text("broken\n", encoding="utf-8")

    code = main(["validate-data", "--path", str(tmp_path), "--json"])
    assert code == 1
    validated = json.loads(capsys.readouterr().out)
    assert validated["ok"] is False
    assert validated["errors"]


def test_info_json(capsys) -> None:
    code = main(["info", "--json"])
    assert code == 0
    info = json.loads(capsys.readouterr().out)
    assert info["version"] == "0.1.0"
    assert info["data_source"] == "mock"
    assert info["adapter"] == "MockDataAdapter"
    assert info["fallback_planned"] is False


def test_invalid_generate_range(tmp_path: Path, capsys) -> None:
    code = main(
        [
            "generate-data",
            "--start",
            "2024-06-30",
            "--end",
            "2024-01-01",
            "--out",
            str(tmp_path),
        ]
    )
    assert code == 1
    assert "生成失败" in capsys.readouterr().err


def test_module_entrypoint() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "future.cli", "info", "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["adapter"] == "MockDataAdapter"
