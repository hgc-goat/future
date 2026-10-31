"""一键运行阶段 1–4，生成 Web 页面所需的全部产物。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from future.backtest import BacktestConfig, run_backtest, write_backtest
from future.data.mock import MockDataAdapter
from future.data.storage import write_dataset
from future.drift import DriftConfig, compute_drift, write_drift_outputs
from future.factors import FactorConfig, compute_factors, write_factors
from future.ml import (
    ModelConfig,
    ModelUnavailableError,
    train_advanced_model,
    write_advanced_outputs,
)
from future.news import (
    NewsConfig,
    analyze_news_increment,
    compute_news_fusion,
    write_news_outputs,
)


@dataclass(frozen=True)
class PipelineConfig:
    """一次完整流水线的参数。"""

    root: Path = Path()
    seed: int = 42
    start: date = date(2020, 1, 2)
    end: date = date(2025, 12, 31)
    train_start: date = date(2021, 1, 1)
    train_end: date = date(2023, 12, 31)
    oos_start: date = date(2024, 1, 1)
    oos_end: date = date(2025, 12, 31)
    factor_start: date = date(2021, 1, 1)
    sector_limit: int | None = None

    def __post_init__(self) -> None:
        if not self.start < self.end:
            raise ValueError("start 必须早于 end")
        if not self.train_start < self.train_end < self.oos_start < self.oos_end:
            raise ValueError("要求 train_start < train_end < oos_start < oos_end")

    @property
    def data_dir(self) -> Path:
        return self.root / "data" / "mock"

    @property
    def factors_dir(self) -> Path:
        return self.root / "data" / "factors"

    @property
    def news_dir(self) -> Path:
        return self.root / "data" / "factors_news"

    @property
    def backtest_dir(self) -> Path:
        return self.root / "artifacts" / "backtest"

    @property
    def news_analysis_dir(self) -> Path:
        return self.root / "artifacts" / "news"

    @property
    def advanced_dir(self) -> Path:
        return self.root / "artifacts" / "advanced"

    @property
    def drift_dir(self) -> Path:
        return self.root / "artifacts" / "drift"


def run_pipeline(config: PipelineConfig | None = None) -> dict[str, Any]:
    """运行完整流水线，返回可 JSON 序列化的摘要。"""

    config = config or PipelineConfig()
    dataset = MockDataAdapter(
        seed=config.seed,
        start=config.start,
        end=config.end,
        sector_limit=config.sector_limit,
    ).load()
    write_dataset(dataset, config.data_dir)

    factor_config = FactorConfig()
    factors = compute_factors(dataset, factor_config)
    factors = factors.loc[factors["date"] >= pd.Timestamp(config.factor_start)]
    factor_manifest = write_factors(factors, config.factors_dir, factor_config)

    backtest_config = BacktestConfig(
        train_start=config.train_start,
        train_end=config.train_end,
        oos_start=config.oos_start,
        oos_end=config.oos_end,
    )
    backtest = run_backtest(dataset, factors, backtest_config)
    write_backtest(backtest, config.backtest_dir)

    news_config = NewsConfig()
    fused = compute_news_fusion(factors, dataset.news, news_config)
    news_analysis = analyze_news_increment(
        dataset,
        fused,
        backtest_config,
        news_config,
    )
    write_news_outputs(
        fused,
        news_analysis,
        config.news_dir,
        config.news_analysis_dir,
        news_config,
    )

    advanced_summary: dict[str, Any]
    try:
        advanced = train_advanced_model(
            dataset,
            fused,
            ModelConfig(
                train_start=config.train_start,
                train_end=config.train_end,
                oos_start=config.oos_start,
                oos_end=config.oos_end,
            ),
        )
        write_advanced_outputs(advanced, config.advanced_dir)
        advanced_summary = {
            "skipped": False,
            "baseline_ic20": advanced.metrics["baseline"]["oos_ic"]["composite"]["20"]["mean_ic"],
            "hgb_ic20": advanced.metrics["hgb"]["oos_ic"]["composite"]["20"]["mean_ic"],
            "hgb_long_short": advanced.metrics["hgb"]["long_short"],
        }
    except ModelUnavailableError as exc:
        advanced_summary = {"skipped": True, "reason": str(exc)}

    drift = compute_drift(
        dataset,
        fused,
        DriftConfig(
            train_start=config.train_start,
            train_end=config.train_end,
            oos_start=config.oos_start,
            oos_end=config.oos_end,
        ),
    )
    write_drift_outputs(drift, config.drift_dir)

    return {
        "ok": True,
        "root": str(config.root),
        "seed": config.seed,
        "start": config.start.isoformat(),
        "end": config.end.isoformat(),
        "data_dir": str(config.data_dir),
        "factors_dir": str(config.factors_dir),
        "news_dir": str(config.news_dir),
        "backtest_dir": str(config.backtest_dir),
        "advanced_dir": str(config.advanced_dir),
        "drift_dir": str(config.drift_dir),
        "rows": {
            "prices": int(len(dataset.prices)),
            "factors": int(factor_manifest["rows"]),
        },
        "backtest": {
            "ic20": backtest.metrics["oos_ic"]["composite"]["20"]["mean_ic"],
            "long_short": backtest.metrics["long_short"],
        },
        "news": {
            "coverage": news_analysis["coverage"],
            "delta": news_analysis["delta"],
        },
        "advanced": advanced_summary,
        "drift": {
            "severity": drift.metrics["severity"],
            "latest": drift.metrics["latest"],
        },
    }


__all__ = ["PipelineConfig", "run_pipeline"]
