from __future__ import annotations

from future.web.data import load_dashboard_data
from future.web.summary import build_summary


def test_summary_is_plain_language(pipeline_artifacts) -> None:
    root, _ = pipeline_artifacts
    data = load_dashboard_data(root)
    summary = build_summary(data)

    assert summary.headline
    assert summary.stance
    assert len(summary.bullets) == 4
    assert len(summary.glossary) >= 8
    assert "不构成投资建议" in summary.caveat
    assert 0.0 <= summary.top_probability <= 100.0
    assert 0.0 <= summary.bottom_probability <= 100.0
    assert summary.top_probability >= summary.bottom_probability
    assert summary.drift_level in {"low", "medium", "high"}

    text = " ".join((summary.headline, *summary.bullets))
    assert "概率" in text
    assert "漂移" in text
    assert "新闻" in text
    assert "回测" in text
