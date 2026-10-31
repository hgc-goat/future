"""新闻推送与板块展望。

算法是“可解释的晚融合”，不是黑盒推荐：
- 新闻：最近 5 个交易日的平均情感、新闻条数、同向确认；
- 财报：最新财务景气因子 z-score；
- 模型：HGB 未来 20 日相对胜率；
- 拥挤度：作为扣分项。
新闻摘要来自 Mock 数据；标题链接到相关新闻搜索，而不是伪造文章 URL。
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote

import numpy as np
import pandas as pd

from future.web.data import DashboardData


@dataclass(frozen=True)
class SectorOutlook:
    sector_id: str
    sector_name: str
    direction: str
    score: float
    confidence: float
    news_score: float
    financial_score: float
    model_score: float
    sentiment: float
    news_count: float
    confirmed: int
    financial: float
    probability: float
    crowding: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class NewsDigest:
    total_items: int
    total_news: int
    average_sentiment: float
    bullish: int
    neutral: int
    bearish: int
    top_sector: str
    top_headline: str
    stance: str
    summary: str


@dataclass(frozen=True)
class NewsItem:
    sector_id: str
    sector_name: str
    headline: str
    summary: str
    url: str
    published_at: str
    sentiment: float
    news_count: int
    direction: str
    confidence: float
    score: float
    verdict: str


def _recent_news(data: DashboardData, window: int = 5) -> pd.DataFrame:
    history = data.history.copy()
    dates = sorted(history["date"].dropna().unique())[-window:]
    recent = history.loc[history["date"].isin(dates)].copy()
    recent["weighted_sentiment"] = recent["sentiment"] * recent["news_count"]
    aggregated = recent.groupby("sector_id").agg(
        news_count=("news_count", "sum"),
        weighted_sentiment=("weighted_sentiment", "sum"),
        confirmed=("news_confirm", lambda values: int((values.abs() > 1e-12).sum())),
    )
    aggregated["sentiment"] = np.where(
        aggregated["news_count"] > 0,
        aggregated["weighted_sentiment"] / aggregated["news_count"],
        0.0,
    )
    return aggregated


def build_outlook(data: DashboardData, sector_id: str | None = None) -> SectorOutlook:
    """把新闻、财报、模型和拥挤度合成可解释的相对方向。"""

    ranking = data.ranking.sort_values("rank").set_index("sector_id")
    selected = sector_id if sector_id in ranking.index else str(ranking.index[0])
    row = ranking.loc[selected]
    recent = _recent_news(data)
    if selected in recent.index:
        news_count = float(recent.loc[selected, "news_count"])
        sentiment = float(recent.loc[selected, "sentiment"])
        confirmed = int(recent.loc[selected, "confirmed"])
    else:
        news_count = 0.0
        sentiment = 0.0
        confirmed = 0

    financial = float(row["financial"])
    probability = float(row["probability"])
    crowding = float(row["crowding"])

    news_score = float(np.clip(50.0 + 50.0 * sentiment, 0.0, 100.0))
    financial_score = float(np.clip(50.0 + 12.0 * financial, 0.0, 100.0))
    model_score = float(np.clip(probability, 0.0, 100.0))
    score = float(
        np.clip(
            0.5 * model_score + 0.3 * news_score + 0.2 * financial_score - 2.0 * max(crowding, 0.0),
            0.0,
            100.0,
        )
    )
    if score >= 58.0:
        direction = "相对偏强"
    elif score <= 42.0:
        direction = "相对偏弱"
    else:
        direction = "中性"

    signals = [np.sign(sentiment), np.sign(financial), np.sign(probability - 50.0)]
    agreement = abs(float(np.sum(signals))) / 3.0
    coverage = min(1.0, news_count / 10.0)
    confidence = float(np.clip(0.35 + 0.3 * agreement + 0.2 * coverage, 0.25, 0.9))

    news_label = "偏正面" if sentiment > 0.15 else "偏负面" if sentiment < -0.15 else "中性"
    financial_label = "偏强" if financial > 0.15 else "偏弱" if financial < -0.15 else "中性"
    crowding_label = "偏高，已扣分" if crowding > 0.5 else "正常"
    reasons = (
        f"新闻：近 5 个交易日 {int(news_count)} 条，平均情感 {sentiment:+.2f}"
        f"（{news_label}，同向确认 {confirmed} 次）。",
        f"财报：最新财务景气因子 z={financial:+.2f}（{financial_label}）。",
        f"模型：HGB 未来 20 日相对胜率 {probability:.1f}%。",
        f"拥挤度：{crowding:+.2f}（{crowding_label}）。",
    )

    return SectorOutlook(
        sector_id=str(selected),
        sector_name=str(row["sector_name"]),
        direction=direction,
        score=score,
        confidence=confidence,
        news_score=news_score,
        financial_score=financial_score,
        model_score=model_score,
        sentiment=sentiment,
        news_count=news_count,
        confirmed=confirmed,
        financial=financial,
        probability=probability,
        crowding=crowding,
        reasons=reasons,
    )


def _search_url(sector_name: str) -> str:
    query = quote(f"{sector_name} 最新 新闻 财报")
    return f"https://so.eastmoney.com/news/s?keyword={query}"


def _headline(outlook: SectorOutlook) -> str:
    sentiment = outlook.sentiment
    direction = outlook.direction
    if outlook.news_count == 0:
        return f"{outlook.sector_name}：消息面安静，等待财报或新催化"
    if sentiment >= 0.15 and direction == "相对偏强":
        return f"{outlook.sector_name}：消息面偏暖，财报与模型形成同向验证"
    if sentiment <= -0.15 and direction == "相对偏弱":
        return f"{outlook.sector_name}：消息面偏冷，短期相对判断承压"
    if sentiment >= 0.15:
        return f"{outlook.sector_name}：消息偏暖，但模型与财报尚未形成合力"
    if sentiment <= -0.15:
        return f"{outlook.sector_name}：消息有分歧，模型仍等待更多确认"
    return f"{outlook.sector_name}：消息面平稳，等待资金流或财报给出方向"


def build_news_digest(items: tuple[NewsItem, ...]) -> NewsDigest:
    """总结今日新闻推送的共同方向与覆盖度。"""

    if not items:
        return NewsDigest(
            total_items=0,
            total_news=0,
            average_sentiment=0.0,
            bullish=0,
            neutral=0,
            bearish=0,
            top_sector="—",
            top_headline="暂无新闻推送",
            stance="暂无足够新闻",
            summary="当前没有可用的新闻样本，等待新数据。",
        )

    total_news = int(sum(item.news_count for item in items))
    weighted = sum(item.sentiment * item.news_count for item in items)
    average_sentiment = weighted / total_news if total_news else 0.0
    bullish = sum(item.direction == "相对偏强" for item in items)
    bearish = sum(item.direction == "相对偏弱" for item in items)
    neutral = len(items) - bullish - bearish
    top = max(items, key=lambda item: (item.news_count, item.score))

    if bullish >= neutral + bearish and bullish >= max(1, len(items) // 2):
        stance = "整体偏暖，但只作同向确认"
    elif bearish >= bullish and bearish >= max(1, len(items) // 2):
        stance = "整体偏冷，先降低信任"
    else:
        stance = "方向分化，等待更多确认"

    summary = (
        f"{len(items)} 条推送中 {bullish} 条偏强、{neutral} 条中性、{bearish} 条偏弱；"
        f"近 5 日共 {total_news} 条新闻，平均情感 {average_sentiment:+.2f}。"
        f"最受关注的是{top.sector_name}。新闻只做同向确认，不单独触发。"
    )
    return NewsDigest(
        total_items=len(items),
        total_news=total_news,
        average_sentiment=average_sentiment,
        bullish=bullish,
        neutral=neutral,
        bearish=bearish,
        top_sector=top.sector_name,
        top_headline=top.headline,
        stance=stance,
        summary=summary,
    )


def build_news_feed(data: DashboardData, limit: int = 5) -> tuple[NewsItem, ...]:
    """按新闻活跃度和情感强度挑选可解释的推送条目。"""

    if limit <= 0:
        return ()
    ranking = data.ranking.sort_values("rank")
    recent = _recent_news(data)
    outlooks = {
        str(row["sector_id"]): build_outlook(data, str(row["sector_id"]))
        for _, row in ranking.iterrows()
    }

    scored: list[tuple[float, str]] = []
    for sector_id, outlook in outlooks.items():
        count = 0.0
        if sector_id in recent.index:
            count = float(recent.loc[sector_id, "news_count"])
        activity = abs(outlook.sentiment) * np.log1p(count) + 0.01 * outlook.score
        scored.append((float(activity), sector_id))
    scored.sort(reverse=True)
    selected_ids = [sector_id for _, sector_id in scored[:limit]]
    if len(selected_ids) < limit:
        for row in ranking.itertuples(index=False):
            sector_id = str(row.sector_id)
            if sector_id not in selected_ids:
                selected_ids.append(sector_id)
            if len(selected_ids) >= limit:
                break

    items: list[NewsItem] = []
    for sector_id in selected_ids:
        outlook = outlooks[sector_id]
        count = int(outlook.news_count)
        summary = (
            f"近 5 个交易日共 {count} 条新闻，平均情感 {outlook.sentiment:+.2f}；"
            f"财务景气因子 z={outlook.financial:+.2f}；"
            f"模型相对胜率 {outlook.probability:.1f}%。"
        )
        verdict = f"{outlook.direction} · 置信度 {outlook.confidence:.0%}"
        items.append(
            NewsItem(
                sector_id=sector_id,
                sector_name=outlook.sector_name,
                headline=_headline(outlook),
                summary=summary,
                url=_search_url(outlook.sector_name),
                published_at=data.latest_date.strftime("%Y-%m-%d"),
                sentiment=outlook.sentiment,
                news_count=count,
                direction=outlook.direction,
                confidence=outlook.confidence,
                score=outlook.score,
                verdict=verdict,
            )
        )
    return tuple(items)


__all__ = [
    "NewsDigest",
    "NewsItem",
    "SectorOutlook",
    "build_news_digest",
    "build_news_feed",
    "build_outlook",
]
