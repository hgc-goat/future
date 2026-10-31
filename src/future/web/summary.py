"""面向非金融读者的页面总结：一句话结论、阅读顺序与术语速查。"""

from __future__ import annotations

from dataclasses import dataclass

from future.web.data import DashboardData

DRIFT_LEVEL_CN = {"low": "低", "medium": "中", "high": "高"}
FEATURE_CN = {
    "momentum": "动量",
    "financial": "财务景气",
    "flow": "资金流",
    "crowding": "拥挤度",
    "sentiment": "新闻情感",
}


@dataclass(frozen=True)
class Summary:
    """页面顶部“一分钟读懂”所需内容。"""

    headline: str
    stance: str
    bullets: tuple[str, ...]
    glossary: tuple[tuple[str, str], ...]
    caveat: str
    top_sector: str
    top_probability: float
    bottom_sector: str
    bottom_probability: float
    drift_level: str
    drift_feature: str


def _mean_ic(metrics: dict, model: str) -> float:
    value = metrics[model]["oos_ic"]["composite"]["20"]["mean_ic"]
    return 0.0 if value is None else float(value)


def build_summary(data: DashboardData) -> Summary:
    """把当前数据翻译成非金融读者也能理解的说明。"""

    ranking = data.ranking.sort_values("rank")
    top = ranking.iloc[0]
    bottom = ranking.iloc[-1]
    top_probability = float(top["probability"])
    bottom_probability = float(bottom["probability"])
    spread = top_probability - bottom_probability

    if top_probability < 55.0:
        stance = "信号偏弱 · 先观察"
        headline = (
            f"今天没有特别突出的板块：最高的{top['sector_name']}也只有"
            f"{top_probability:.1f}%的相对胜率。"
        )
    elif spread < 12.0:
        stance = "信号分散 · 头部差距不大"
        headline = (
            f"模型略微偏向{top['sector_name']}，但它和后面的板块差距不大，"
            "更适合观察而不是重仓判断。"
        )
    else:
        stance = "相对明确 · 仍不是涨跌保证"
        headline = (
            f"{top['sector_name']}是今天模型最偏好的板块，但{top_probability:.1f}%只代表相对胜率。"
        )

    severity = data.drift["severity"]
    level = str(severity["level"])
    level_cn = DRIFT_LEVEL_CN.get(level, level)
    psi = data.drift["psi"]["by_feature"]
    max_feature = max(psi, key=psi.get)
    feature_cn = FEATURE_CN.get(max_feature, max_feature)

    hgb_ic = _mean_ic(data.advanced, "hgb")
    linear_ic = _mean_ic(data.advanced, "baseline")
    confirmed_rate = float((data.history["news_confirm"].abs() > 1e-12).mean())

    bullets = (
        (
            f"先看概率：最高是{top['sector_name']} {top_probability:.1f}%，"
            f"最低是{bottom['sector_name']} {bottom_probability:.1f}%。"
            "50% 约等于抛硬币，65% 也意味着约三分之一概率会跑输板块中位数。"
        ),
        (
            f"再看漂移：当前{level_cn}（{float(severity['overall']):.2f}），"
            f"最近的分布变化主要来自{feature_cn}。"
            "它提醒模型依赖的规律可能在变，不是买卖信号。"
        ),
        (
            f"新闻只做同向确认：约{confirmed_rate:.1%}的历史样本获得了确认加成。"
            "新闻不足或方向相反时，系统不会单独行动。"
        ),
        (
            f"最后看回测：进阶模型的历史排序区分力（IC）是 {hgb_ic:.3f}，"
            f"线性融合是 {linear_ic:.3f}。"
            "这只说明历史上有一些区分度，不能直接外推成未来收益。"
        ),
    )

    glossary = (
        ("相对胜率", "未来 20 个交易日跑赢同一天所有板块中位数的模型估计，不是涨跌预测。"),
        ("动量", "最近 20/60/120 个交易日涨得是否稳；涨得稳的板块可能延续，也可能追高。"),
        ("财务景气", "盈利、收入、ROE 等基本面是否在改善，数据有披露延迟。"),
        ("资金流", "主力或北向资金是否持续流入，反映短期关注度。"),
        ("拥挤度", "换手、成交和波动是否过热；越拥挤，系统越谨慎并扣分。"),
        ("新闻确认", "只有当新闻数量足够多、情感方向与基础信号一致时才加分。"),
        ("漂移", "训练期的规律在最新数据里是否发生变化；越高越需要降低信任。"),
        ("IC", "历史预测排名和实际排名的相关性，不是收益率；0.05–0.10 已算有信息但噪声很大。"),
        ("多空", "同时做多模型最看好的层、做空最不看好的层，用来观察排序有没有区分度。"),
    )

    return Summary(
        headline=headline,
        stance=stance,
        bullets=bullets,
        glossary=glossary,
        caveat="这是 Mock 数据的相对排序演示，不构成投资建议；概率、回测和漂移都不能保证未来收益。",
        top_sector=str(top["sector_name"]),
        top_probability=top_probability,
        bottom_sector=str(bottom["sector_name"]),
        bottom_probability=bottom_probability,
        drift_level=level,
        drift_feature=feature_cn,
    )


__all__ = ["Summary", "build_summary"]
