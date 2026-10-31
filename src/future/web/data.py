"""Web 展示所需的数据装载与派生指标。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from future.data.base import DataSourceError


@dataclass
class DashboardData:
    """一次页面加载所需的全量数据。"""

    ranking: pd.DataFrame
    history: pd.DataFrame
    sectors: pd.DataFrame
    advanced: dict[str, Any]
    backtest: dict[str, Any]
    drift: dict[str, Any]
    rolling_ic: pd.DataFrame
    latest_date: pd.Timestamp
    probability_source: str


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DataSourceError(f"缺少文件：{path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataSourceError(f"读取失败：{path}：{exc}") from exc
    if not isinstance(payload, dict):
        raise DataSourceError(f"JSON 顶层必须为对象：{path}")
    return payload


def _sigmoid(value: pd.Series) -> pd.Series:
    return 1.0 / (1.0 + np.exp(-value.astype(float)))


def load_dashboard_data(root: Path | str = Path()) -> DashboardData:
    """读取阶段 2–4 产物，构建最新排名和历史序列。"""

    root = Path(root)
    data_dir = root / "data" / "mock"
    fused_path = root / "data" / "factors_news" / "fused_factors.csv"
    predictions_path = root / "artifacts" / "advanced" / "predictions.csv"
    advanced_path = root / "artifacts" / "advanced" / "metrics.json"
    backtest_path = root / "artifacts" / "backtest" / "metrics.json"
    drift_path = root / "artifacts" / "drift" / "drift.json"
    rolling_ic_path = root / "artifacts" / "drift" / "rolling_ic.csv"

    for path in (data_dir, fused_path, predictions_path, advanced_path, backtest_path):
        if not path.exists():
            raise DataSourceError(
                f"缺少页面数据：{path}。请先依次运行阶段 1–4 的生成/因子/回测/新闻/模型/漂移命令。"
            )

    try:
        sectors = pd.read_csv(data_dir / "sectors.csv", dtype={"sector_id": "string"})
        prices = pd.read_csv(
            data_dir / "prices.csv",
            parse_dates=["date"],
            dtype={"sector_id": "string"},
        )
        fused = pd.read_csv(fused_path, parse_dates=["date"], dtype={"sector_id": "string"})
        predictions = pd.read_csv(
            predictions_path,
            parse_dates=["date"],
            dtype={"sector_id": "string"},
        )
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise DataSourceError(f"页面数据读取失败：{exc}") from exc

    if fused.empty or predictions.empty:
        raise DataSourceError("因子或模型预测为空，无法构建页面")

    latest_date = fused["date"].max()
    latest = fused.loc[fused["date"] == latest_date].copy()
    latest_predictions = predictions.loc[
        predictions["date"] == latest_date,
        ["sector_id", "p_up"],
    ].copy()

    prices = prices.sort_values(["sector_id", "date"])
    prices["ret_20d"] = prices.groupby("sector_id")["close"].pct_change(
        20,
        fill_method=None,
    )
    latest_prices = (
        prices.groupby("sector_id", as_index=False)
        .tail(1)[["sector_id", "close", "ret_20d"]]
        .copy()
    )

    ranking = latest.merge(latest_predictions, on="sector_id", how="left")
    ranking = ranking.merge(
        sectors[["sector_id", "sector_name"]],
        on="sector_id",
        how="left",
    )
    ranking = ranking.merge(latest_prices, on="sector_id", how="left")
    fallback = _sigmoid(ranking["composite_fused"])
    probability_source = "HGB"
    if ranking["p_up"].isna().any():
        probability_source = "HGB + 线性映射补空"
        ranking["p_up"] = ranking["p_up"].fillna(fallback)
    ranking["linear_score"] = ranking["composite_fused"]
    ranking["probability"] = ranking["p_up"] * 100.0
    ranking["ret_20d"] = ranking["ret_20d"] * 100.0
    ranking["sector_name"] = ranking["sector_name"].fillna(ranking["sector_id"])
    ranking["rank"] = ranking["probability"].rank(method="first", ascending=False).astype(int)
    ranking = ranking.sort_values("rank").reset_index(drop=True)

    history = fused.merge(
        predictions[["date", "sector_id", "p_up"]],
        on=["date", "sector_id"],
        how="left",
    )
    history = history.merge(
        prices[["date", "sector_id", "close"]],
        on=["date", "sector_id"],
        how="left",
    )
    history = history.sort_values(["date", "sector_id"]).reset_index(drop=True)

    advanced = _read_json(advanced_path)
    backtest = _read_json(backtest_path)
    drift = _read_json(drift_path)
    if not rolling_ic_path.is_file():
        rolling_ic = pd.DataFrame(columns=["date", "ic", "rolling_ic"])
    else:
        rolling_ic = pd.read_csv(rolling_ic_path, parse_dates=["date"])

    return DashboardData(
        ranking=ranking,
        history=history,
        sectors=sectors,
        advanced=advanced,
        backtest=backtest,
        drift=drift,
        rolling_ic=rolling_ic,
        latest_date=latest_date,
        probability_source=probability_source,
    )


__all__ = ["DashboardData", "load_dashboard_data"]
