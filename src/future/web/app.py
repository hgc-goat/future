"""板块趋势概率评估系统的 Dash 页面。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dash_table, dcc, html

from future.data.base import DataSourceError
from future.web.data import DashboardData, load_dashboard_data
from future.web.newsfeed import (
    NewsDigest,
    NewsItem,
    SectorOutlook,
    build_news_digest,
    build_news_feed,
    build_outlook,
)
from future.web.summary import build_daily_digest, build_summary

ASSETS_DIR = Path(__file__).parent / "assets"

PAGE = "#090e13"
SURFACE = "#111821"
SURFACE_2 = "#18222c"
TEXT = "#f5f7f4"
SECONDARY = "#b8c4c0"
MUTED = "#7e8c88"
GRID = "#24313a"
BASELINE = "#35444c"
BORDER = "rgba(255,255,255,0.11)"
BLUE = "#3987e5"
AQUA = "#199e70"
YELLOW = "#d9a441"
RED = "#e5725b"
GOOD = "#0ca30c"
WARNING = "#fab219"
CRITICAL = "#d03b3b"

PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}
FONT_FAMILY = 'system-ui, -apple-system, "Segoe UI", sans-serif'
LEVEL_LABEL = {"low": "低", "medium": "中", "high": "高"}
LEVEL_COLOR = {"low": GOOD, "medium": WARNING, "high": CRITICAL}


def _base_layout(fig: go.Figure, height: int, *, hovermode: str = "closest") -> go.Figure:
    fig.update_layout(
        template=None,
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": FONT_FAMILY, "color": SECONDARY, "size": 12},
        margin={"l": 8, "r": 12, "t": 8, "b": 28},
        hovermode=hovermode,
        showlegend=False,
        hoverlabel={
            "bgcolor": SURFACE_2,
            "bordercolor": BORDER,
            "font": {"color": TEXT, "family": FONT_FAMILY},
        },
    )
    fig.update_xaxes(
        gridcolor=GRID,
        zerolinecolor=BASELINE,
        linecolor=BASELINE,
        tickfont={"color": MUTED, "size": 11},
        title_font={"color": MUTED, "size": 11},
        automargin=True,
    )
    fig.update_yaxes(
        gridcolor=GRID,
        zerolinecolor=BASELINE,
        linecolor=BASELINE,
        tickfont={"color": MUTED, "size": 11},
        title_font={"color": MUTED, "size": 11},
        automargin=True,
    )
    return fig


def _fmt_pct(value: float | None, digits: int = 1) -> str:
    return "—" if value is None else f"{value:.{digits}%}"


def _fmt_num(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _kpi(label: str, value: str, detail: str, accent: str = BLUE) -> html.Div:
    return html.Div(
        className="kpi-card",
        children=[
            html.Div(className="kpi-accent", style={"background": accent}),
            html.P(label, className="kpi-label"),
            html.Div(value, className="kpi-value"),
            html.P(detail, className="kpi-detail"),
        ],
    )


def _chip(label: str, value: str, accent: str = BLUE) -> html.Div:
    return html.Div(
        className="chip",
        children=[
            html.Span(label, className="chip-label"),
            html.Span(value, className="chip-value", style={"color": accent}),
        ],
    )


def _memo_section(summary) -> html.Section:
    glossary_items = [
        html.Div(
            className="glossary-item",
            children=[
                html.Span(term, className="glossary-term"),
                html.Span(description, className="glossary-copy"),
            ],
        )
        for term, description in summary.glossary
    ]
    return html.Section(
        id="page-daily-memo",
        className="memo",
        children=[
            html.Div(
                className="memo-top",
                children=[
                    html.Div(
                        className="memo-main",
                        children=[
                            html.Div(
                                className="memo-kicker",
                                children=[
                                    html.Span("BRIEFING"),
                                    html.Span(
                                        "给第一次看的人",
                                        className="memo-kicker-note",
                                    ),
                                ],
                            ),
                            html.H2(summary.headline, className="memo-headline"),
                            html.P(summary.stance, className="memo-stance"),
                            html.P(
                                "它不预测绝对涨跌，只比较板块之间的相对强弱。",
                                className="memo-scope",
                            ),
                            html.Ul(
                                [html.Li(text) for text in summary.bullets],
                                className="memo-bullets",
                            ),
                        ],
                    ),
                    html.Aside(
                        className="memo-side",
                        children=[
                            html.Div(
                                className="memo-stamp",
                                children=[
                                    html.Span("MOCK"),
                                    html.Strong("20D"),
                                    html.Span("相对胜率"),
                                ],
                            ),
                            html.Div(
                                className="memo-principles",
                                children=[
                                    html.P("读数顺序", className="memo-side-title"),
                                    html.Ol(
                                        [
                                            html.Li("先看最高概率有没有明显高于 50%"),
                                            html.Li("再看漂移是否升到中/高"),
                                            html.Li("最后看回测能不能区分强弱"),
                                        ],
                                        className="memo-steps",
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            html.Details(
                className="memo-glossary",
                open=False,
                children=[
                    html.Summary("术语速查 · 第一次看先读这 9 个词"),
                    html.Div(className="glossary-grid", children=glossary_items),
                ],
            ),
            html.P(summary.caveat, className="memo-caveat"),
        ],
    )


def _direction_color(direction: str) -> str:
    if direction == "相对偏强":
        return GOOD
    if direction == "相对偏弱":
        return CRITICAL
    return WARNING


def _news_item_article(item: NewsItem) -> html.Article:
    color = _direction_color(item.direction)
    sentiment_label = (
        "偏正面" if item.sentiment > 0.15 else "偏负面" if item.sentiment < -0.15 else "中性"
    )
    return html.Article(
        className="news-item",
        children=[
            html.Div(
                className="news-meta",
                children=[
                    html.Span("MOCK NEWS", className="news-source"),
                    html.Span(item.published_at, className="news-date"),
                    html.Span(item.sector_name, className="news-sector"),
                    html.Span(sentiment_label, style={"color": color}),
                ],
            ),
            html.A(
                item.headline,
                href=item.url,
                target="_blank",
                rel="noreferrer",
                className="news-title",
            ),
            html.P(item.summary, className="news-summary"),
            html.Div(
                className="news-footer",
                children=[
                    html.Span(
                        item.verdict,
                        className="news-verdict",
                        style={"color": color, "borderColor": color},
                    ),
                    html.A(
                        "搜索相关报道 ↗",
                        href=item.url,
                        target="_blank",
                        rel="noreferrer",
                        className="news-link",
                    ),
                ],
            ),
        ],
    )


def _outlook_component(
    label: str,
    value: float,
    detail: str,
    color: str,
) -> html.Div:
    return html.Div(
        className="outlook-component",
        children=[
            html.Div(
                className="outlook-component-head",
                children=[
                    html.Span(label),
                    html.Span(f"{value:.0f}", style={"color": color}),
                ],
            ),
            html.Div(
                className="outlook-track",
                children=[
                    html.Div(
                        className="outlook-fill",
                        style={"width": f"{value:.0f}%", "background": color},
                    )
                ],
            ),
            html.P(detail, className="outlook-component-note"),
        ],
    )


def _outlook_components(outlook: SectorOutlook) -> list[Any]:
    return [
        _outlook_component(
            "新闻",
            outlook.news_score,
            f"情感 {outlook.sentiment:+.2f} · {int(outlook.news_count)} 条",
            BLUE,
        ),
        _outlook_component(
            "财报",
            outlook.financial_score,
            f"财务因子 z={outlook.financial:+.2f}",
            AQUA,
        ),
        _outlook_component(
            "模型",
            outlook.model_score,
            f"20 日相对胜率 {outlook.probability:.1f}%",
            YELLOW,
        ),
        html.Div(
            className="outlook-crowding",
            children=[
                html.Span("拥挤度扣分"),
                html.Span(f"{outlook.crowding:+.2f}"),
            ],
        ),
    ]


def _outlook_proxy_note(target_type: str) -> str:
    if target_type == "板块指数":
        return "当前是板块级相对方向，不是个股或基金的买卖建议。"
    if target_type == "跟踪基金 / ETF（代理）":
        return "当前 Mock 数据没有基金持仓和净值，使用同板块信号代理；不代表具体基金代码。"
    return "当前 Mock 数据没有公司级财报与公告，使用同板块信号代理；不代表具体个股。"


def _outlook_card(outlook: SectorOutlook) -> html.Section:
    return html.Section(
        className="card outlook-card",
        children=[
            html.Div(
                className="card-head",
                children=[
                    html.Div(
                        [
                            html.P("OUTLOOK / 板块展望", className="eyebrow"),
                            html.H2("未来 20 天怎么看"),
                        ]
                    ),
                    html.Span("相对方向 · 不是涨跌预测", className="card-hint"),
                ],
            ),
            html.Div(
                className="outlook-head",
                children=[
                    html.Div(
                        [
                            html.P("当前关注对象", className="eyebrow"),
                            html.H3(id="outlook-sector", children=outlook.sector_name),
                            html.P(
                                id="outlook-meta",
                                className="outlook-meta",
                                children="板块级信号 · 在主排序表中选择其他板块会同步更新",
                            ),
                        ]
                    ),
                    html.Div(
                        className="outlook-score-block",
                        children=[
                            html.Div(
                                id="outlook-score",
                                className="outlook-score",
                                children=f"{outlook.score:.0f}",
                            ),
                            html.P("综合方向分 / 100", className="outlook-score-label"),
                            html.Div(
                                id="outlook-direction",
                                className="outlook-direction",
                                style={
                                    "color": _direction_color(outlook.direction),
                                    "borderColor": _direction_color(outlook.direction),
                                },
                                children=outlook.direction,
                            ),
                            html.P(
                                id="outlook-confidence",
                                className="outlook-confidence",
                                children=f"置信度 {outlook.confidence:.0%}",
                            ),
                        ],
                    ),
                ],
            ),
            dcc.RadioItems(
                id="target-type",
                options=[
                    {"label": "板块指数", "value": "板块指数"},
                    {"label": "跟踪基金 / ETF（代理）", "value": "跟踪基金 / ETF（代理）"},
                    {"label": "成分股（代理）", "value": "成分股（代理）"},
                ],
                value="板块指数",
                inline=True,
                className="target-switch",
                inputStyle={"marginRight": "5px", "accentColor": BLUE},
                labelStyle={"marginRight": "14px"},
            ),
            html.Div(
                id="outlook-components",
                className="outlook-components",
                children=_outlook_components(outlook),
            ),
            html.Ul(
                id="outlook-reasons",
                className="outlook-reasons",
                children=[html.Li(reason) for reason in outlook.reasons],
            ),
            html.P(
                "方向分 = 50% 模型 + 30% 新闻 + 20% 财报 − 拥挤度扣分；"
                "置信度看三路信号是否同向、新闻覆盖是否足够。",
                className="outlook-formula",
            ),
            html.P(
                id="outlook-proxy-note",
                className="outlook-proxy-note",
                children=_outlook_proxy_note("板块指数"),
            ),
        ],
    )


def _market_figure(ranking: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Histogram(
            x=ranking["probability"],
            nbinsx=10,
            marker={"color": BLUE, "line": {"width": 0}},
            opacity=0.86,
            hovertemplate="概率 %{x:.1f}%<br>%{y} 个板块<extra></extra>",
        )
    )
    _base_layout(fig, 220, hovermode="x")
    fig.update_xaxes(title_text="20 日相对胜率 (%)", range=[0, 100])
    fig.update_yaxes(title_text="板块数", gridcolor=GRID)
    fig.add_vline(x=50, line={"color": WARNING, "width": 1})
    fig.add_annotation(
        x=50,
        y=1.06,
        yref="paper",
        text="50% 抛硬币线",
        showarrow=False,
        font={"color": WARNING, "size": 10},
    )
    return fig


def _daily_stat(label: str, value: str, detail: str) -> html.Div:
    return html.Div(
        className="daily-stat",
        children=[
            html.P(label, className="daily-stat-label"),
            html.Div(value, className="daily-stat-value"),
            html.P(detail, className="daily-stat-detail"),
        ],
    )


def _daily_summary_section(
    data: DashboardData,
    digest,
    news_items: tuple[NewsItem, ...],
) -> html.Section:
    top_news = news_items[0] if news_items else None
    color = (
        GOOD
        if digest.news_bullish > digest.news_bearish
        else CRITICAL
        if digest.news_bearish > digest.news_bullish
        else WARNING
    )
    return html.Section(
        id="page-daily-summary",
        className="daily-grid",
        children=[
            html.Section(
                className="card daily-market-card",
                children=[
                    html.Div(
                        className="card-head",
                        children=[
                            html.Div(
                                [
                                    html.P("MARKET / 今日大盘", className="eyebrow"),
                                    html.H2("板块温度"),
                                ]
                            ),
                            html.Span(digest.market_label, className="badge"),
                        ],
                    ),
                    html.Div(
                        className="daily-market-head",
                        children=[
                            _daily_stat(
                                "相对胜率 ≥ 50%",
                                f"{digest.breadth:.0f}%",
                                "高于抛硬币线的板块占比",
                            ),
                            _daily_stat(
                                "平均相对胜率",
                                f"{digest.average_probability:.1f}%",
                                "全部板块横截面均值",
                            ),
                            _daily_stat(
                                "头部 - 尾部",
                                f"{digest.top_probability - digest.bottom_probability:.1f}pp",
                                f"{digest.top_sector} vs {digest.bottom_sector}",
                            ),
                        ],
                    ),
                    dcc.Graph(
                        figure=_market_figure(data.ranking),
                        config=PLOTLY_CONFIG,
                        style={"height": "220px"},
                    ),
                    html.Div(
                        className="daily-foot",
                        children=[
                            _chip(
                                "头部",
                                f"{digest.top_sector} {digest.top_probability:.1f}%",
                                AQUA,
                            ),
                            _chip(
                                "尾部",
                                f"{digest.bottom_sector} {digest.bottom_probability:.1f}%",
                                MUTED,
                            ),
                        ],
                    ),
                    html.P(
                        "用板块横截面代理大盘温度，不是指数涨跌预测。",
                        className="note",
                    ),
                ],
            ),
            html.Section(
                className="card daily-news-card",
                children=[
                    html.Div(
                        className="card-head",
                        children=[
                            html.Div(
                                [
                                    html.P("NEWS / 新闻预测总结", className="eyebrow"),
                                    html.H2("今天新闻怎么说"),
                                ]
                            ),
                            html.Span(
                                f"近 5 日 · {len(news_items)} 条推送",
                                className="card-hint",
                            ),
                        ],
                    ),
                    html.Div(
                        className="daily-news-stats",
                        children=[
                            _daily_stat("新闻总量", str(digest.news_count), "近 5 个交易日"),
                            _daily_stat(
                                "平均情感",
                                f"{digest.news_sentiment:+.2f}",
                                "按新闻条数加权",
                            ),
                            _daily_stat(
                                "偏强 / 偏弱",
                                f"{digest.news_bullish} / {digest.news_bearish}",
                                f"中性 {digest.news_neutral}",
                            ),
                        ],
                    ),
                    html.Div(
                        className="daily-top-news",
                        children=[
                            html.P("最值得看的一条", className="eyebrow"),
                            html.A(
                                top_news.headline if top_news else "暂无新闻推送",
                                href=top_news.url if top_news else "#",
                                target="_blank",
                                rel="noreferrer",
                                className="news-title",
                            ),
                            html.P(
                                top_news.summary if top_news else "新闻样本不足。",
                                className="news-summary",
                            ),
                            html.Span(
                                top_news.verdict if top_news else "观察",
                                className="news-verdict",
                                style={"color": color, "borderColor": color},
                            ),
                        ],
                    ),
                    html.Div(
                        className="daily-news-more",
                        children=[
                            html.P("其他值得看", className="eyebrow"),
                            *[
                                html.Div(
                                    className="daily-news-row",
                                    children=[
                                        html.Span(
                                            item.sector_name,
                                            className="daily-news-sector",
                                        ),
                                        html.A(
                                            item.headline,
                                            href=item.url,
                                            target="_blank",
                                            rel="noreferrer",
                                            className="daily-news-title",
                                        ),
                                        html.Span(
                                            item.direction,
                                            style={"color": _direction_color(item.direction)},
                                        ),
                                    ],
                                )
                                for item in news_items[1:4]
                            ],
                        ],
                    ),
                    html.P(
                        "完整推送、新闻链接和板块代理展望在“新闻”页。",
                        className="note",
                    ),
                ],
            ),
        ],
    )


def _news_summary_banner(digest: NewsDigest) -> html.Div:
    color = GOOD if digest.bullish >= digest.bearish else CRITICAL
    return html.Div(
        className="news-summary-banner",
        children=[
            html.Div(
                className="news-summary-copy",
                children=[
                    html.P("NEWS DIGEST / 新闻总结", className="eyebrow"),
                    html.H2(digest.stance),
                    html.P(digest.summary, className="news-summary-text"),
                ],
            ),
            html.Div(
                className="news-summary-stats",
                children=[
                    _daily_stat(
                        "推送分布",
                        f"{digest.bullish} / {digest.neutral} / {digest.bearish}",
                        "偏强 / 中性 / 偏弱",
                    ),
                    _daily_stat(
                        "平均情感",
                        f"{digest.average_sentiment:+.2f}",
                        f"近 5 日 {digest.total_news} 条新闻",
                    ),
                    _daily_stat(
                        "最受关注",
                        digest.top_sector,
                        "新闻活跃度最高",
                    ),
                ],
            ),
            html.Span("同向确认", className="news-summary-seal", style={"borderColor": color}),
        ],
    )


def _news_section(
    items: tuple[NewsItem, ...],
    outlook: SectorOutlook,
    digest: NewsDigest,
) -> html.Section:
    return html.Section(
        id="page-news",
        className="news-grid",
        style={"display": "none"},
        children=[
            _news_summary_banner(digest),
            html.Section(
                className="card news-card",
                children=[
                    html.Div(
                        className="card-head",
                        children=[
                            html.Div(
                                [
                                    html.P("NEWS WIRE / 今日推送", className="eyebrow"),
                                    html.H2("新闻与判断"),
                                ]
                            ),
                            html.Span("模拟摘要 · 标题跳转相关搜索", className="card-hint"),
                        ],
                    ),
                    html.P(
                        "这些是根据 Mock 新闻情感、财报因子和模型概率生成的简报，不是实时抓取；"
                        "标题会打开东方财富的相关新闻搜索。",
                        className="card-lede",
                    ),
                    html.Div(
                        className="news-list",
                        children=[_news_item_article(item) for item in items],
                    ),
                ],
            ),
            _outlook_card(outlook),
        ],
    )


def _factor_contributions(row: pd.Series) -> list[tuple[str, float]]:
    return [
        ("动量", float(row["momentum"]) / 3.0),
        ("财务景气", float(row["financial"]) / 3.0),
        ("资金流", float(row["flow"]) / 3.0),
        ("拥挤度(反向)", -0.5 * float(row["crowding"])),
        ("新闻确认", 0.15 * float(row["news_confirm"])),
    ]


def _attribution_figure(row: pd.Series) -> go.Figure:
    contributions = _factor_contributions(row)
    labels = [item[0] for item in contributions]
    values = [item[1] for item in contributions]
    colors = [BLUE if value >= 0 else RED for value in values]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker={"color": colors, "line": {"width": 0}},
            text=[f"{value:+.2f}" for value in values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:+.3f}<extra></extra>",
        )
    )
    _base_layout(fig, 250)
    fig.update_xaxes(
        title_text="对 composite_fused 的线性贡献",
        zerolinecolor=SECONDARY,
        zerolinewidth=1,
        range=[
            min(values + [0.0]) - 0.2,
            max(values + [0.0]) + 0.2,
        ],
    )
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)", automargin=True)
    return fig


def _price_figure(history: pd.DataFrame, sector_name: str) -> go.Figure:
    if history.empty:
        fig = go.Figure()
        _base_layout(fig, 230)
        fig.add_annotation(
            text="暂无价格数据",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"color": MUTED},
        )
        return fig

    frame = history.sort_values("date").tail(252)
    fig = go.Figure(
        go.Scatter(
            x=frame["date"],
            y=frame["close"],
            mode="lines",
            name="收盘价",
            line={"color": BLUE, "width": 2},
            hovertemplate="%{x|%Y-%m-%d}<br>收盘 %{y:.2f} 元<extra></extra>",
        )
    )
    last = frame.iloc[-1]
    fig.add_trace(
        go.Scatter(
            x=[last["date"]],
            y=[last["close"]],
            mode="markers",
            marker={"color": BLUE, "size": 8, "line": {"color": SURFACE, "width": 2}},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_annotation(
        x=last["date"],
        y=last["close"],
        text=f"{last['close']:.2f}",
        showarrow=False,
        xshift=-8,
        yshift=14,
        font={"color": SECONDARY, "size": 11},
    )
    _base_layout(fig, 230)
    fig.update_yaxes(title_text="元", gridcolor=GRID)
    fig.update_xaxes(showgrid=False)
    return fig


def _drift_gauge(drift: dict[str, Any]) -> go.Figure:
    severity = drift["severity"]
    overall = float(severity["overall"])
    level = severity["level"]
    color = LEVEL_COLOR.get(level, WARNING)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=overall,
            number={"font": {"color": TEXT, "size": 38}, "valueformat": ".2f"},
            title={
                "text": f"综合严重度 · {LEVEL_LABEL.get(level, level)}",
                "font": {"color": MUTED, "size": 12},
            },
            gauge={
                "axis": {
                    "range": [0, 1],
                    "tickvals": [0, 0.3, 0.6, 1],
                    "ticktext": ["0", "0.3", "0.6", "1"],
                    "tickcolor": MUTED,
                    "tickfont": {"color": MUTED, "size": 10},
                },
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": SURFACE_2,
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 0.3], "color": "rgba(12,163,12,0.16)"},
                    {"range": [0.3, 0.6], "color": "rgba(250,178,25,0.16)"},
                    {"range": [0.6, 1.0], "color": "rgba(208,59,59,0.16)"},
                ],
            },
        )
    )
    fig.update_layout(
        template=None,
        height=230,
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": FONT_FAMILY},
        margin={"l": 18, "r": 18, "t": 30, "b": 8},
    )
    return fig


def _psi_figure(drift: dict[str, Any]) -> go.Figure:
    labels_map = {
        "momentum": "动量",
        "financial": "财务景气",
        "flow": "资金流",
        "crowding": "拥挤度",
        "sentiment": "新闻情感",
    }
    psi = drift["psi"]["by_feature"]
    items = sorted(psi.items(), key=lambda item: item[1])
    labels = [labels_map.get(key, key) for key, _ in items]
    values = [float(value) for _, value in items]
    max_value = max(values) if values else 0.0
    colors = [WARNING if value == max_value and value >= 0.25 else BLUE for value in values]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker={"color": colors, "line": {"width": 0}},
            text=[f"{value:.3f}" for value in values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y} PSI: %{x:.3f}<extra></extra>",
        )
    )
    _base_layout(fig, 230)
    fig.update_xaxes(title_text="分布变化（越高越需谨慎）", range=[0, max(0.5, max_value * 1.35)])
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)", automargin=True)
    fig.add_vline(x=0.25, line={"color": MUTED, "width": 1})
    fig.add_annotation(
        x=0.25,
        y=1.06,
        yref="paper",
        text="0.25",
        showarrow=False,
        font={"color": MUTED, "size": 10},
    )
    return fig


def _rolling_ic_figure(rolling_ic: pd.DataFrame, drift: dict[str, Any]) -> go.Figure:
    frame = rolling_ic.dropna(subset=["rolling_ic"]).copy()
    fig = go.Figure()
    if not frame.empty:
        fig.add_trace(
            go.Scatter(
                x=frame["date"],
                y=frame["rolling_ic"],
                mode="lines",
                name="滚动 IC",
                line={"color": BLUE, "width": 2},
                hovertemplate="%{x|%Y-%m-%d}<br>滚动 IC %{y:.4f}<extra></extra>",
            )
        )
        last = frame.iloc[-1]
        fig.add_trace(
            go.Scatter(
                x=[last["date"]],
                y=[last["rolling_ic"]],
                mode="markers",
                marker={"color": BLUE, "size": 8, "line": {"color": SURFACE, "width": 2}},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_annotation(
            x=last["date"],
            y=last["rolling_ic"],
            text=f"{last['rolling_ic']:.3f}",
            showarrow=False,
            xshift=-10,
            yshift=16,
            font={"color": SECONDARY, "size": 11},
        )
    train_mean = float(drift["train"]["ic_mean"])
    fig.add_hline(y=0.0, line={"color": BASELINE, "width": 1})
    fig.add_hline(
        y=train_mean,
        line={"color": MUTED, "width": 1},
        annotation_text=f"训练均值 {train_mean:.3f}",
        annotation_position="bottom right",
        annotation_font={"color": MUTED, "size": 10},
    )
    _base_layout(fig, 230, hovermode="x unified")
    fig.update_yaxes(title_text="20 日滚动 IC", gridcolor=GRID)
    fig.update_xaxes(showgrid=False)
    return fig


def _layer_figure(advanced: dict[str, Any]) -> go.Figure:
    layers = [str(index) for index in range(1, 6)]
    baseline = advanced["baseline"]["layers"]
    hgb = advanced["hgb"]["layers"]
    x = [f"L{index}" for index in layers]
    baseline_y = [float(baseline[layer]["annualized_return"]) * 100.0 for layer in layers]
    hgb_y = [float(hgb[layer]["annualized_return"]) * 100.0 for layer in layers]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="线性融合",
            x=x,
            y=baseline_y,
            marker={"color": BLUE, "line": {"width": 0}},
            hovertemplate="%{x} 线性<br>年化 %{y:.1f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="HGB",
            x=x,
            y=hgb_y,
            marker={"color": AQUA, "line": {"width": 0}},
            hovertemplate="%{x} HGB<br>年化 %{y:.1f}%<extra></extra>",
        )
    )
    _base_layout(fig, 260)
    fig.update_layout(
        barmode="group",
        bargap=0.28,
        bargroupgap=0.12,
        showlegend=True,
        legend={
            "orientation": "h",
            "x": 0,
            "y": 1.12,
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"color": SECONDARY, "size": 11},
        },
        margin={"l": 8, "r": 12, "t": 34, "b": 28},
    )
    fig.update_yaxes(title_text="年化 %", gridcolor=GRID, zerolinecolor=BASELINE)
    fig.update_xaxes(showgrid=False)
    return fig


def _metrics_table(advanced: dict[str, Any]) -> html.Table:
    baseline = advanced["baseline"]
    hgb = advanced["hgb"]

    def row(label: str, base_value: str, hgb_value: str) -> html.Tr:
        return html.Tr(
            [
                html.Td(label),
                html.Td(base_value, className="num"),
                html.Td(hgb_value, className="num"),
            ]
        )

    return html.Table(
        className="metric-table",
        children=[
            html.Thead(
                html.Tr(
                    [
                        html.Th("指标"),
                        html.Th("线性融合"),
                        html.Th("HGB"),
                    ]
                )
            ),
            html.Tbody(
                [
                    row(
                        "5 日 OOS IC",
                        _fmt_num(baseline["oos_ic"]["composite"]["5"]["mean_ic"], 4),
                        _fmt_num(hgb["oos_ic"]["composite"]["5"]["mean_ic"], 4),
                    ),
                    row(
                        "20 日 OOS IC",
                        _fmt_num(baseline["oos_ic"]["composite"]["20"]["mean_ic"], 4),
                        _fmt_num(hgb["oos_ic"]["composite"]["20"]["mean_ic"], 4),
                    ),
                    row(
                        "20 日 ICIR",
                        _fmt_num(baseline["oos_ic"]["composite"]["20"]["icir"], 3),
                        _fmt_num(hgb["oos_ic"]["composite"]["20"]["icir"], 3),
                    ),
                    row(
                        "多空年化",
                        _fmt_pct(baseline["long_short"]["annualized_return"]),
                        _fmt_pct(hgb["long_short"]["annualized_return"]),
                    ),
                    row(
                        "多空 Sharpe",
                        _fmt_num(baseline["long_short"]["sharpe"]),
                        _fmt_num(hgb["long_short"]["sharpe"]),
                    ),
                    row(
                        "最大回撤",
                        _fmt_pct(baseline["long_short"]["max_drawdown"]),
                        _fmt_pct(hgb["long_short"]["max_drawdown"]),
                    ),
                ]
            ),
        ],
    )


def _detail_stats(row: pd.Series) -> list[html.Div]:
    return [
        _chip("模型概率", f"{row['probability']:.1f}%", BLUE),
        _chip("线性分数", f"{row['linear_score']:+.2f}", AQUA),
        _chip("20 日涨跌", f"{row['ret_20d']:+.1f}%", GOOD if row["ret_20d"] >= 0 else RED),
        _chip("新闻确认", f"{row['news_confirm']:+.2f}", YELLOW if row["news_confirm"] else MUTED),
    ]


def _factor_grid(row: pd.Series) -> list[html.Div]:
    items = [
        ("动量", f"{row['momentum']:+.2f}"),
        ("财务景气", f"{row['financial']:+.2f}"),
        ("资金流", f"{row['flow']:+.2f}"),
        ("拥挤度", f"{row['crowding']:+.2f}"),
        ("新闻情感", f"{row['sentiment']:+.2f}"),
        ("新闻数量", f"{int(row['news_count'])}"),
    ]
    return [
        html.Div(
            className="factor-item",
            children=[
                html.Span(label, className="factor-label"),
                html.Span(value, className="factor-value"),
            ],
        )
        for label, value in items
    ]


def create_app(data: DashboardData | None = None, root: Path | str = Path()) -> Dash:
    """创建并配置 Dash 应用。"""

    data = data or load_dashboard_data(root)
    ranking = data.ranking.copy()
    ranking["sector_id"] = ranking["sector_id"].astype(str)
    ranking["sector_name"] = ranking["sector_name"].astype(str)
    ranking_index = ranking.set_index("sector_id")
    default_sector = str(ranking.iloc[0]["sector_id"])
    default_row = ranking_index.loc[default_sector]
    default_history = data.history.loc[data.history["sector_id"] == default_sector]

    table_columns = [
        {"name": "排名", "id": "rank", "type": "numeric"},
        {"name": "板块", "id": "sector_name", "type": "text"},
        {
            "name": "20 日胜率 %",
            "id": "probability",
            "type": "numeric",
            "format": {"specifier": ".1f"},
        },
        {
            "name": "线性分数",
            "id": "linear_score",
            "type": "numeric",
            "format": {"specifier": "+.2f"},
        },
        {
            "name": "20 日涨跌 %",
            "id": "ret_20d",
            "type": "numeric",
            "format": {"specifier": "+.1f"},
        },
        {
            "name": "新闻确认",
            "id": "news_confirm",
            "type": "numeric",
            "format": {"specifier": "+.2f"},
        },
    ]
    table_data = ranking[
        ["rank", "sector_name", "probability", "linear_score", "ret_20d", "news_confirm"]
    ].to_dict("records")
    rank_to_sector = {
        int(rank): str(sector_id)
        for rank, sector_id in zip(ranking["rank"], ranking["sector_id"], strict=True)
    }
    sector_options = [
        {
            "label": f"{row['sector_name']} · {row['probability']:.1f}%",
            "value": row["sector_id"],
        }
        for _, row in ranking.iterrows()
    ]

    advanced = data.advanced
    baseline_ic = float(advanced["baseline"]["oos_ic"]["composite"]["20"]["mean_ic"])
    hgb_ic = float(advanced["hgb"]["oos_ic"]["composite"]["20"]["mean_ic"])
    hgb_long_short = advanced["hgb"]["long_short"]
    drift = data.drift
    severity = drift["severity"]
    severity_level = LEVEL_LABEL.get(severity["level"], severity["level"])
    top = ranking.iloc[0]
    summary = build_summary(data)
    news_items = build_news_feed(data, limit=5)
    news_digest = build_news_digest(news_items)
    daily_digest = build_daily_digest(data, news_items)
    initial_outlook = build_outlook(data, default_sector)

    app = Dash(
        __name__,
        assets_folder=str(ASSETS_DIR),
        title="future | 板块趋势概率",
        update_title=None,
        meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    )

    app.layout = html.Div(
        className="app-shell",
        children=[
            html.Header(
                className="topbar",
                children=[
                    html.Div(
                        className="brand-block",
                        children=[
                            html.Div("F", className="brand-mark"),
                            html.Div(
                                [
                                    html.P("FUTURE / SIGNAL MEMO", className="eyebrow"),
                                    html.H1("板块趋势概率"),
                                    html.P(
                                        "先读人话结论，再看证据。",
                                        className="subtitle",
                                    ),
                                ]
                            ),
                        ],
                    ),
                    html.Div(
                        className="date-chip",
                        children=[
                            html.Span("数据截止"),
                            html.Strong(data.latest_date.strftime("%Y-%m-%d")),
                            html.Span("· 20D 相对胜率 · Mock", className="muted-inline"),
                        ],
                    ),
                ],
            ),
            html.Nav(
                className="page-nav",
                children=[
                    html.Button(
                        "日报",
                        id="nav-daily",
                        n_clicks=0,
                        type="button",
                        className="nav-item nav-item--active",
                    ),
                    html.Button(
                        "板块",
                        id="nav-sectors",
                        n_clicks=0,
                        type="button",
                        className="nav-item",
                    ),
                    html.Button(
                        "新闻",
                        id="nav-news",
                        n_clicks=0,
                        type="button",
                        className="nav-item",
                    ),
                    html.Button(
                        "模型健康",
                        id="nav-health",
                        n_clicks=0,
                        type="button",
                        className="nav-item",
                    ),
                ],
            ),
            _memo_section(summary),
            html.Section(
                id="page-daily-kpi",
                className="kpi-grid",
                children=[
                    _kpi(
                        "模型最偏好的板块",
                        f"{top['probability']:.1f}%",
                        f"{top['sector_name']} · 线性分数 {top['linear_score']:+.2f}",
                        BLUE,
                    ),
                    _kpi(
                        "历史预测力 IC",
                        f"{hgb_ic:.3f}",
                        f"线性融合 {baseline_ic:.3f} · 越高越能区分强弱",
                        AQUA,
                    ),
                    _kpi(
                        "历史多空年化",
                        _fmt_pct(hgb_long_short["annualized_return"]),
                        f"Sharpe {hgb_long_short['sharpe']:.2f} · "
                        f"最大回撤 {_fmt_pct(hgb_long_short['max_drawdown'])}",
                        AQUA,
                    ),
                    _kpi(
                        "漂移预警",
                        f"{severity['overall']:.2f}",
                        f"{severity_level} · 最大变化：{summary.drift_feature}",
                        LEVEL_COLOR.get(severity["level"], WARNING),
                    ),
                ],
            ),
            _daily_summary_section(data, daily_digest, news_items),
            html.Section(
                id="page-sectors",
                className="dashboard-grid",
                style={"display": "none"},
                children=[
                    html.Section(
                        className="card ranking-card",
                        children=[
                            html.Div(
                                className="card-head",
                                children=[
                                    html.Div(
                                        [
                                            html.P("RANKING", className="eyebrow"),
                                            html.H2("板块概率排序"),
                                        ]
                                    ),
                                    html.Span(
                                        "可排序 · 可搜索 · 点一行看它为什么被选中",
                                        className="card-hint",
                                    ),
                                ],
                            ),
                            html.Div(
                                className="ranking-toolbar",
                                children=[
                                    dcc.Dropdown(
                                        id="sector-select",
                                        options=sector_options,
                                        value=default_sector,
                                        clearable=False,
                                        searchable=True,
                                        placeholder="选择板块",
                                        className="sector-select",
                                    ),
                                    dcc.Input(
                                        id="sector-search",
                                        type="search",
                                        placeholder="搜索板块…",
                                        className="sector-search",
                                    ),
                                    html.Span(
                                        "概率是 20 日相对胜率，不是涨跌幅",
                                        className="muted-inline",
                                    ),
                                ],
                            ),
                            dash_table.DataTable(
                                id="ranking-table",
                                columns=table_columns,
                                data=table_data,
                                sort_action="native",
                                filter_action="none",
                                page_action="native",
                                page_size=12,
                                row_selectable="single",
                                active_cell=None,
                                style_as_list_view=True,
                                style_table={"minWidth": "100%", "overflowX": "auto"},
                                style_header={
                                    "backgroundColor": "rgba(0,0,0,0)",
                                    "color": MUTED,
                                    "fontWeight": "600",
                                    "fontSize": "11px",
                                    "letterSpacing": "0.08em",
                                    "textTransform": "uppercase",
                                    "borderBottom": f"1px solid {GRID}",
                                    "padding": "10px 10px",
                                },
                                style_filter={
                                    "backgroundColor": SURFACE,
                                    "color": SECONDARY,
                                    "borderBottom": f"1px solid {GRID}",
                                },
                                style_cell={
                                    "backgroundColor": "rgba(0,0,0,0)",
                                    "color": TEXT,
                                    "border": "none",
                                    "borderBottom": f"1px solid {GRID}",
                                    "padding": "9px 10px",
                                    "fontFamily": FONT_FAMILY,
                                    "fontSize": "13px",
                                    "textAlign": "left",
                                    "minWidth": "80px",
                                },
                                style_cell_conditional=[
                                    {"if": {"column_id": "rank"}, "textAlign": "right"},
                                    {"if": {"column_id": "probability"}, "textAlign": "right"},
                                    {"if": {"column_id": "linear_score"}, "textAlign": "right"},
                                    {"if": {"column_id": "ret_20d"}, "textAlign": "right"},
                                    {"if": {"column_id": "news_confirm"}, "textAlign": "right"},
                                ],
                                style_data={
                                    "fontVariantNumeric": "tabular-nums",
                                },
                                style_data_conditional=[
                                    {
                                        "if": {"row_index": "odd"},
                                        "backgroundColor": "rgba(255,255,255,0.012)",
                                    },
                                    {
                                        "if": {"column_id": "probability"},
                                        "color": AQUA,
                                        "fontWeight": "600",
                                    },
                                    {
                                        "if": {"column_id": "rank"},
                                        "color": MUTED,
                                    },
                                    {
                                        "if": {"state": "selected"},
                                        "backgroundColor": "rgba(57, 135, 229, 0.18)",
                                        "color": TEXT,
                                    },
                                    {
                                        "if": {"state": "active"},
                                        "backgroundColor": "rgba(57, 135, 229, 0.12)",
                                        "color": TEXT,
                                    },
                                ],
                            ),
                        ],
                    ),
                    html.Section(
                        className="card detail-card",
                        children=[
                            html.Div(
                                className="detail-head",
                                children=[
                                    html.Div(
                                        [
                                            html.P(
                                                "SELECTED SECTOR / 选中板块",
                                                className="eyebrow",
                                            ),
                                            html.H2(
                                                id="detail-name",
                                                children=default_row["sector_name"],
                                            ),
                                            html.P(id="detail-meta", className="detail-meta"),
                                        ]
                                    ),
                                    html.Div(
                                        id="detail-probability",
                                        className="detail-probability",
                                        children=f"{default_row['probability']:.1f}%",
                                    ),
                                ],
                            ),
                            html.Div(
                                id="detail-stats",
                                className="chip-row",
                                children=_detail_stats(default_row),
                            ),
                            html.Div(
                                className="chart-block",
                                children=[
                                    html.Div(
                                        className="chart-caption",
                                        children="它为什么被选中 · 线性贡献",
                                    ),
                                    dcc.Graph(
                                        id="attribution-chart",
                                        figure=_attribution_figure(default_row),
                                        config=PLOTLY_CONFIG,
                                        style={"height": "250px"},
                                    ),
                                ],
                            ),
                            html.Div(
                                className="chart-block",
                                children=[
                                    html.Div(
                                        className="chart-caption",
                                        children="价格走势 · 近 252 个交易日",
                                    ),
                                    dcc.Graph(
                                        id="price-chart",
                                        figure=_price_figure(
                                            default_history, default_row["sector_name"]
                                        ),
                                        config=PLOTLY_CONFIG,
                                        style={"height": "230px"},
                                    ),
                                ],
                            ),
                            html.Div(
                                id="factor-grid",
                                className="factor-grid",
                                children=_factor_grid(default_row),
                            ),
                            html.P(
                                "线性贡献只解释 composite_fused；"
                                "HGB 概率来自分类模型，无法直接拆成因子贡献。",
                                className="note",
                            ),
                        ],
                    ),
                ],
            ),
            _news_section(news_items, initial_outlook, news_digest),
            html.Section(
                id="page-health",
                className="bottom-grid",
                style={"display": "none"},
                children=[
                    html.Section(
                        className="card drift-card",
                        children=[
                            html.Div(
                                className="card-head",
                                children=[
                                    html.Div(
                                        [
                                            html.P("SIGNAL HEALTH / 漂移监控", className="eyebrow"),
                                            html.H2("模型还靠谱吗？"),
                                        ]
                                    ),
                                    html.Span(
                                        f"{severity_level} · {severity['overall']:.2f}",
                                        className="badge",
                                        style={
                                            "color": LEVEL_COLOR.get(severity["level"], WARNING),
                                            "borderColor": LEVEL_COLOR.get(
                                                severity["level"], WARNING
                                            ),
                                        },
                                    ),
                                ],
                            ),
                            html.P(
                                "分数越高，越应该把排序当作参考而不是结论。"
                                f"当前最大变化来自{summary.drift_feature}；这是模型健康度，不是涨跌信号。",
                                className="card-lede",
                            ),
                            html.Div(
                                className="drift-grid",
                                children=[
                                    dcc.Graph(
                                        figure=_drift_gauge(drift),
                                        config=PLOTLY_CONFIG,
                                        className="drift-gauge",
                                        style={"height": "230px"},
                                    ),
                                    dcc.Graph(
                                        figure=_psi_figure(drift),
                                        config=PLOTLY_CONFIG,
                                        className="drift-psi",
                                        style={"height": "230px"},
                                    ),
                                ],
                            ),
                            dcc.Graph(
                                figure=_rolling_ic_figure(data.rolling_ic, drift),
                                config=PLOTLY_CONFIG,
                                className="drift-ic",
                                style={"height": "230px"},
                            ),
                            html.Div(
                                className="chip-row compact",
                                children=[
                                    _chip("预测力变化", f"{severity['ic']:.2f}", BLUE),
                                    _chip(
                                        "数据分布变化", f"{severity['distribution']:.2f}", WARNING
                                    ),
                                    _chip(
                                        "最新滚动 IC",
                                        f"{drift['latest']['rolling_ic']:.4f}"
                                        if drift["latest"]["rolling_ic"] is not None
                                        else "—",
                                        SECONDARY,
                                    ),
                                ],
                            ),
                        ],
                    ),
                    html.Section(
                        className="card backtest-card",
                        children=[
                            html.Div(
                                className="card-head",
                                children=[
                                    html.Div(
                                        [
                                            html.P("BACKTEST / 回测说明", className="eyebrow"),
                                            html.H2("历史表现，不是预测"),
                                        ]
                                    ),
                                    html.Span("2024–2025 · 未计成本", className="card-hint"),
                                ],
                            ),
                            html.P(
                                "IC 看模型有没有区分强弱的能力；多空年化看排序在历史样本里的收益。"
                                "它们都不是未来收益承诺。",
                                className="card-lede",
                            ),
                            html.Div(
                                className="compare-grid",
                                children=[
                                    html.Div(
                                        className="compare-card",
                                        children=[
                                            html.P("线性融合 20 日 IC", className="kpi-label"),
                                            html.Div(
                                                f"{baseline_ic:.4f}",
                                                className="compare-value",
                                            ),
                                            html.P(
                                                "多空年化 "
                                                f"{_fmt_pct(advanced['baseline']['long_short']['annualized_return'])}",
                                                className="kpi-detail",
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="compare-card",
                                        children=[
                                            html.P("HGB 20 日 IC", className="kpi-label"),
                                            html.Div(
                                                f"{hgb_ic:.4f}", className="compare-value accent"
                                            ),
                                            html.P(
                                                "多空年化 "
                                                f"{_fmt_pct(hgb_long_short['annualized_return'])}",
                                                className="kpi-detail",
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                            dcc.Graph(
                                figure=_layer_figure(advanced),
                                config=PLOTLY_CONFIG,
                                style={"height": "260px"},
                            ),
                            _metrics_table(advanced),
                            html.P(
                                "图例：线性融合 = 阶段 3 composite_fused；HGB = 阶段 4 分类模型。",
                                className="note",
                            ),
                        ],
                    ),
                ],
            ),
            html.Footer(
                className="footer",
                children="future · Mock 数据展示 · 不构成投资建议",
            ),
            dcc.Store(id="selected-sector", data=default_sector),
        ],
    )

    def _switch_page(*_clicks: int) -> tuple[Any, ...]:
        active = (ctx.triggered_id or "nav-daily").replace("nav-", "")
        if active not in {"daily", "sectors", "news", "health"}:
            active = "daily"
        section_ids = (
            "page-daily-memo",
            "page-daily-kpi",
            "page-daily-summary",
            "page-sectors",
            "page-news",
            "page-health",
        )
        visible = {
            "daily": {"page-daily-memo", "page-daily-kpi", "page-daily-summary"},
            "sectors": {"page-sectors"},
            "news": {"page-news"},
            "health": {"page-health"},
        }[active]
        display_types = {
            "page-daily-memo": "block",
            "page-daily-kpi": "grid",
            "page-daily-summary": "grid",
            "page-sectors": "grid",
            "page-news": "grid",
            "page-health": "grid",
        }
        styles = [
            {"display": display_types[section_id]} if section_id in visible else {"display": "none"}
            for section_id in section_ids
        ]
        nav_classes = [
            "nav-item nav-item--active" if active == "daily" else "nav-item",
            "nav-item nav-item--active" if active == "sectors" else "nav-item",
            "nav-item nav-item--active" if active == "news" else "nav-item",
            "nav-item nav-item--active" if active == "health" else "nav-item",
        ]
        return (*styles, *nav_classes)

    app.callback(
        Output("page-daily-memo", "style"),
        Output("page-daily-kpi", "style"),
        Output("page-daily-summary", "style"),
        Output("page-sectors", "style"),
        Output("page-news", "style"),
        Output("page-health", "style"),
        Output("nav-daily", "className"),
        Output("nav-sectors", "className"),
        Output("nav-news", "className"),
        Output("nav-health", "className"),
        Input("nav-daily", "n_clicks"),
        Input("nav-sectors", "n_clicks"),
        Input("nav-news", "n_clicks"),
        Input("nav-health", "n_clicks"),
    )(_switch_page)

    def _select_sector(
        active_cell: dict[str, Any] | None,
        virtual_data: list[dict[str, Any]] | None,
        selected_rows: list[int] | None,
        virtual_selected_rows: list[int] | None,
        dropdown_value: str | None,
        current: str | None,
    ) -> tuple[str, str]:
        trigger = ctx.triggered_id
        if trigger == "sector-select" and dropdown_value:
            return dropdown_value, dropdown_value
        row_index: int | None = None
        if virtual_selected_rows:
            row_index = virtual_selected_rows[0]
        elif selected_rows:
            row_index = selected_rows[0]
        elif active_cell:
            row = active_cell.get("row")
            if isinstance(row, int):
                row_index = row
        if virtual_data and row_index is not None and 0 <= row_index < len(virtual_data):
            row = virtual_data[row_index]
            if row.get("rank") is not None:
                sector_id = rank_to_sector.get(int(row["rank"]))
            else:
                sector_id = None
            if sector_id in ranking_index.index:
                return str(sector_id), str(sector_id)
        selected = current if current in ranking_index.index else default_sector
        return selected, selected

    app.callback(
        Output("selected-sector", "data"),
        Output("sector-select", "value"),
        Input("ranking-table", "active_cell"),
        Input("ranking-table", "derived_virtual_data"),
        Input("ranking-table", "selected_rows"),
        Input("ranking-table", "derived_virtual_selected_rows"),
        Input("sector-select", "value"),
        State("selected-sector", "data"),
    )(_select_sector)

    def _filter_table(query: str | None) -> list[dict[str, Any]]:
        if not query:
            return table_data
        normalized = query.strip().lower()
        if not normalized:
            return table_data
        return [row for row in table_data if normalized in str(row["sector_name"]).lower()]

    app.callback(
        Output("ranking-table", "data"),
        Input("sector-search", "value"),
    )(_filter_table)

    def _update_outlook(
        sector_id: str | None,
        target_type: str | None,
    ) -> tuple[Any, ...]:
        outlook = build_outlook(data, sector_id)
        color = _direction_color(outlook.direction)
        meta = (
            f"HGB 相对胜率 {outlook.probability:.1f}% · "
            f"财务 z={outlook.financial:+.2f} · "
            f"5 日新闻 {int(outlook.news_count)} 条"
        )
        return (
            outlook.sector_name,
            meta,
            f"{outlook.score:.0f}",
            outlook.direction,
            {"color": color, "borderColor": color},
            f"置信度 {outlook.confidence:.0%}",
            _outlook_components(outlook),
            [html.Li(reason) for reason in outlook.reasons],
            _outlook_proxy_note(target_type or "板块指数"),
        )

    app.callback(
        Output("outlook-sector", "children"),
        Output("outlook-meta", "children"),
        Output("outlook-score", "children"),
        Output("outlook-direction", "children"),
        Output("outlook-direction", "style"),
        Output("outlook-confidence", "children"),
        Output("outlook-components", "children"),
        Output("outlook-reasons", "children"),
        Output("outlook-proxy-note", "children"),
        Input("selected-sector", "data"),
        Input("target-type", "value"),
    )(_update_outlook)

    def _update_detail(sector_id: str | None) -> tuple[Any, ...]:
        selected = sector_id if sector_id in ranking_index.index else default_sector
        row = ranking_index.loc[selected]
        history = data.history.loc[data.history["sector_id"] == selected]
        meta = (
            f"排名 #{int(row['rank'])} · 线性分数 {float(row['linear_score']):+.2f} · "
            f"20 日涨跌 {float(row['ret_20d']):+.1f}%"
        )
        return (
            row["sector_name"],
            meta,
            f"{float(row['probability']):.1f}%",
            _detail_stats(row),
            _attribution_figure(row),
            _price_figure(history, str(row["sector_name"])),
            _factor_grid(row),
        )

    app.callback(
        Output("detail-name", "children"),
        Output("detail-meta", "children"),
        Output("detail-probability", "children"),
        Output("detail-stats", "children"),
        Output("attribution-chart", "figure"),
        Output("price-chart", "figure"),
        Output("factor-grid", "children"),
        Input("selected-sector", "data"),
    )(_update_detail)

    return app


def run_app(
    host: str = "127.0.0.1",
    port: int = 8050,
    debug: bool = False,
    data_root: Path | str = Path(),
) -> None:
    """启动 Dash 服务。"""

    data = load_dashboard_data(data_root)
    app = create_app(data=data)
    app.run(host=host, port=port, debug=debug, use_reloader=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="启动 future Dash 页面")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--data-root", type=Path, default=Path())
    args = parser.parse_args(argv)
    try:
        run_app(
            host=args.host,
            port=args.port,
            debug=args.debug,
            data_root=args.data_root,
        )
    except DataSourceError as exc:
        print(f"页面数据加载失败：{exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["create_app", "main", "run_app"]
