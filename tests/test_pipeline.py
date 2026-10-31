from __future__ import annotations

from pathlib import Path

from future.web.data import load_dashboard_data


def _assert_file(root: Path, relative: str) -> None:
    assert (root / relative).is_file(), relative


def test_pipeline_generates_all_artifacts(pipeline_artifacts) -> None:
    root, result = pipeline_artifacts

    assert result["ok"] is True
    assert result["rows"]["prices"] > 0
    assert result["rows"]["factors"] > 0
    assert result["backtest"]["ic20"] is not None

    for relative in (
        "data/mock/manifest.json",
        "data/mock/prices.csv",
        "data/factors/factors.csv",
        "data/factors_news/fused_factors.csv",
        "artifacts/backtest/metrics.json",
        "artifacts/news/analysis.json",
        "artifacts/drift/drift.json",
    ):
        _assert_file(root, relative)

    if not result["advanced"]["skipped"]:
        _assert_file(root, "artifacts/advanced/metrics.json")
        _assert_file(root, "artifacts/advanced/predictions.csv")
        assert result["advanced"]["hgb_ic20"] is not None

    assert result["news"]["coverage"]["total_rows"] > 0
    assert 0.0 <= result["drift"]["severity"]["overall"] <= 1.0


def test_pipeline_dashboard_data(pipeline_artifacts) -> None:
    root, _ = pipeline_artifacts
    data = load_dashboard_data(root)

    assert not data.ranking.empty
    assert len(data.ranking) == 12
    assert data.ranking["probability"].between(0.0, 100.0).all()
    assert data.history["close"].notna().any()
    assert data.advanced
    assert data.drift["severity"]["level"] in {"low", "medium", "high"}
