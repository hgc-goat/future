from __future__ import annotations

from future.web.data import load_dashboard_data
from future.web.newsfeed import build_news_feed
from future.web.summary import build_daily_digest, build_summary


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


def test_daily_digest_summarizes_market_and_news(pipeline_artifacts) -> None:
    root, _ = pipeline_artifacts
    data = load_dashboard_data(root)
    items = build_news_feed(data, limit=5)
    digest = build_daily_digest(data, items)

    assert 0.0 <= digest.breadth <= 100.0
    assert 0.0 <= digest.average_probability <= 100.0
    assert digest.top_probability >= digest.bottom_probability
    assert digest.news_count > 0
    assert digest.news_bullish + digest.news_neutral + digest.news_bearish == len(items)
    assert digest.top_news_headline
    assert digest.market_label in {"板块整体偏暖", "板块整体偏冷", "板块分化"}
