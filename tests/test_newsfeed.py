from __future__ import annotations

from future.web.data import load_dashboard_data
from future.web.newsfeed import (
    build_deepseek_prompt,
    build_news_digest,
    build_news_feed,
    build_outlook,
)


def test_news_feed_items(pipeline_artifacts) -> None:
    root, _ = pipeline_artifacts
    data = load_dashboard_data(root)
    items = build_news_feed(data, limit=5)

    assert len(items) == 5
    for item in items:
        assert item.sector_name
        assert item.headline
        assert item.summary
        assert item.url.startswith("https://")
        assert item.direction in {"相对偏强", "中性", "相对偏弱"}
        assert 0.0 <= item.confidence <= 1.0
        assert 0.0 <= item.score <= 100.0


def test_outlook_combines_news_financials_and_model(pipeline_artifacts) -> None:
    root, _ = pipeline_artifacts
    data = load_dashboard_data(root)
    top_sector = str(data.ranking.sort_values("rank").iloc[0]["sector_id"])
    outlook = build_outlook(data, top_sector)

    assert outlook.sector_id == top_sector
    assert 0.0 <= outlook.score <= 100.0
    assert 0.0 <= outlook.news_score <= 100.0
    assert 0.0 <= outlook.financial_score <= 100.0
    assert 0.0 <= outlook.model_score <= 100.0
    assert len(outlook.reasons) == 4
    assert "新闻" in outlook.reasons[0]
    assert "财报" in outlook.reasons[1]
    assert "模型" in outlook.reasons[2]


def test_news_digest_summarizes_directions(pipeline_artifacts) -> None:
    root, _ = pipeline_artifacts
    data = load_dashboard_data(root)
    items = build_news_feed(data, limit=5)
    digest = build_news_digest(items)

    assert digest.total_items == 5
    assert digest.bullish + digest.neutral + digest.bearish == 5
    assert digest.total_news > 0
    assert digest.top_sector
    assert "同向确认" in digest.summary
    assert digest.stance


def test_deepseek_prompt_covers_news_and_fundamentals(pipeline_artifacts) -> None:
    root, _ = pipeline_artifacts
    data = load_dashboard_data(root)
    items = build_news_feed(data, limit=5)
    digest = build_news_digest(items)
    prompt = build_deepseek_prompt(data, digest, items)

    assert "财报" in prompt
    assert "产业新闻" in prompt
    assert "公司公告" in prompt
    assert "来源" in prompt
    assert "未验证" in prompt
    assert "数据是 Mock" in prompt
    assert data.latest_date.strftime("%Y-%m-%d") in prompt
