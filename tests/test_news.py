from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest

from future.data.base import DataSourceError
from future.factors import compute_factors
from future.news import (
    NewsConfig,
    analyze_news_increment,
    compute_news_fusion,
    news_coverage,
    write_news_outputs,
)

DATE = pd.Timestamp("2024-01-02")


def _factor_frame(composites: list[float], sectors: list[str]) -> pd.DataFrame:
    rows = []
    for sector_id, composite in zip(sectors, composites, strict=True):
        row = {
            "date": DATE,
            "sector_id": sector_id,
            "momentum_20": 0.0,
            "momentum_60": 0.0,
            "momentum_120": 0.0,
            "momentum": 0.0,
            "financial": 0.0,
            "flow": 0.0,
            "crowding": 0.0,
            "composite": composite,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _news_frame(sentiments: list[float], counts: list[int], sectors: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": DATE,
            "sector_id": sectors,
            "sentiment": sentiments,
            "news_count": counts,
        }
    )


def test_news_only_confirms_same_direction() -> None:
    sectors = ["a", "b", "c", "d", "e", "f"]
    factors = _factor_frame([1.0, 1.0, 1.0, -1.0, -1.0, 0.0], sectors)
    news = _news_frame([0.5, 0.5, 0.05, 0.5, -0.4, 0.8], [3, 2, 5, 5, 3, 5], sectors)

    fused = compute_news_fusion(factors, news)

    confirm = dict(zip(fused["sector_id"], fused["news_confirm"], strict=True))
    assert confirm["a"] == pytest.approx(0.5)  # 同向且满足阈值
    assert confirm["b"] == 0.0  # 新闻数不足
    assert confirm["c"] == 0.0  # 情感情感绝对值不足
    assert confirm["d"] == 0.0  # 方向相反，不确认
    assert confirm["e"] == pytest.approx(-0.4)  # 同向负情感
    assert confirm["f"] == 0.0  # 基础信号为 0，不确认

    fused_map = dict(zip(fused["sector_id"], fused["composite_fused"], strict=True))
    assert fused_map["a"] == pytest.approx(1.0 + 0.15 * 0.5)
    assert fused_map["e"] == pytest.approx(-1.0 + 0.15 * -0.4)
    assert fused_map["d"] == pytest.approx(-1.0)


def test_news_coverage() -> None:
    sectors = ["a", "b", "c", "d"]
    factors = _factor_frame([1.0, 1.0, -1.0, 0.0], sectors)
    news = _news_frame([0.5, 0.5, -0.4, 0.8], [3, 2, 3, 5], sectors)
    fused = compute_news_fusion(factors, news)

    coverage = news_coverage(fused)

    assert coverage["total_rows"] == 4
    assert coverage["eligible_rows"] == 3
    assert coverage["confirmed_rows"] == 2
    assert coverage["eligible_rate"] == 0.75
    assert coverage["confirmed_rate"] == 0.5


def test_news_config_validation() -> None:
    with pytest.raises(ValueError, match="min_news_count"):
        NewsConfig(min_news_count=-1)
    with pytest.raises(ValueError, match="min_abs_sentiment"):
        NewsConfig(min_abs_sentiment=1.5)


def test_fusion_requires_composite() -> None:
    factors = _factor_frame([1.0], ["a"]).drop(columns=["composite"])
    news = _news_frame([0.5], [3], ["a"])
    with pytest.raises(DataSourceError, match="composite"):
        compute_news_fusion(factors, news)


def test_analysis_schema(stage2_dataset) -> None:
    factors = compute_factors(stage2_dataset)
    fused = compute_news_fusion(factors, stage2_dataset.news)
    analysis = analyze_news_increment(stage2_dataset, fused)

    assert analysis["news_config"] == {
        "min_news_count": 3,
        "min_abs_sentiment": 0.1,
        "boost": 0.15,
    }
    assert set(analysis["coverage"]) == {
        "total_rows",
        "eligible_rows",
        "confirmed_rows",
        "eligible_rate",
        "confirmed_rate",
        "same_sign_rate_among_eligible",
        "avg_confirmed_sentiment",
    }
    assert set(analysis["correlations"]) == {"pearson", "spearman"}
    assert set(analysis["backtest"]) == {"base", "fused"}
    assert set(analysis["delta"]) == {
        "ic_5",
        "ic_20",
        "annualized_return",
        "sharpe",
        "max_drawdown",
        "win_rate",
    }
    assert analysis["backtest"]["base"]["long_short"]["n_days"] > 0
    assert analysis["backtest"]["fused"]["oos_ic"]["composite"]["20"]["mean_ic"] is not None


def test_analysis_reproducible(stage2_dataset) -> None:
    factors = compute_factors(stage2_dataset)
    fused = compute_news_fusion(factors, stage2_dataset.news)
    first = analyze_news_increment(stage2_dataset, fused)
    second = analyze_news_increment(stage2_dataset, fused)
    assert first == second


def test_write_news_outputs(tmp_path, stage2_dataset) -> None:
    factors = compute_factors(stage2_dataset)
    fused = compute_news_fusion(factors, stage2_dataset.news)
    analysis = analyze_news_increment(stage2_dataset, fused)

    write_news_outputs(fused, analysis, tmp_path / "factors", tmp_path / "analysis")

    assert (tmp_path / "factors" / "fused_factors.csv").is_file()
    assert (tmp_path / "factors" / "manifest.json").is_file()
    assert (tmp_path / "analysis" / "analysis.json").is_file()
    assert (tmp_path / "analysis" / "manifest.json").is_file()
    saved = json.loads((tmp_path / "analysis" / "analysis.json").read_text(encoding="utf-8"))
    assert saved["coverage"] == analysis["coverage"]


def test_analysis_uses_oos_config(stage2_dataset) -> None:
    from future.backtest import BacktestConfig

    factors = compute_factors(stage2_dataset)
    fused = compute_news_fusion(factors, stage2_dataset.news)
    config = BacktestConfig(
        train_start=date(2022, 1, 1),
        train_end=date(2023, 12, 31),
        oos_start=date(2024, 1, 1),
        oos_end=date(2024, 12, 31),
    )
    analysis = analyze_news_increment(stage2_dataset, fused, config)
    assert analysis["backtest"]["base"]["config"]["oos_end"] == "2024-12-31"
