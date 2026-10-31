"""结构化板块因子。

三个核心因子：
- momentum: 20/60/120 日动量，横截面 z-score 后等权；
- financial: 最新已披露的 ROE / 营收增速 / 利润增速，横截面 z-score 后合成；
- flow: 20 日主力净流入与北向净流入的相对强度。

crowding（拥挤度）为反向提示，复合因子中给负权重。
因子日期 t 只使用 t 收盘及之前可用的数据；回测在 t+1 才开始使用信号。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from future.data.base import Dataset, DataSourceError

FACTOR_COLUMNS = (
    "date",
    "sector_id",
    "momentum_20",
    "momentum_60",
    "momentum_120",
    "momentum",
    "financial",
    "flow",
    "crowding",
    "composite",
)


@dataclass(frozen=True)
class FactorConfig:
    """因子窗口与复合权重。"""

    momentum_windows: tuple[int, ...] = (20, 60, 120)
    flow_window: int = 20
    crowding_window: int = 5
    crowding_weight: float = 0.5


def _cross_sectional_z(frame: pd.DataFrame, column: str) -> pd.Series:
    grouped = frame.groupby("date")[column]
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0.0, np.nan)
    return (frame[column] - mean) / std


def _momentum(prices: pd.DataFrame, config: FactorConfig) -> pd.DataFrame:
    close_wide = prices.pivot(index="date", columns="sector_id", values="close").sort_index()
    grid = prices[["date", "sector_id"]].drop_duplicates()
    momentum = grid.copy()
    z_columns: list[str] = []

    for window in config.momentum_windows:
        wide = close_wide.pct_change(window, fill_method=None)
        long = wide.stack().rename(f"momentum_{window}").reset_index()
        long["sector_id"] = long["sector_id"].astype(str)
        momentum = momentum.merge(long, on=["date", "sector_id"], how="left")
        z_column = f"momentum_{window}_z"
        momentum[z_column] = _cross_sectional_z(momentum, f"momentum_{window}")
        z_columns.append(z_column)

    momentum["momentum"] = momentum[z_columns].mean(axis=1)
    return momentum[["date", "sector_id", "momentum_20", "momentum_60", "momentum_120", "momentum"]]


def _financial(dataset: Dataset) -> pd.DataFrame:
    grid = dataset.prices[["date", "sector_id"]].drop_duplicates()
    grid = grid.assign(sector_id=grid["sector_id"].astype(str))
    daily = grid.sort_values(["sector_id", "date"])
    fundamentals = dataset.fundamentals.copy()
    fundamentals["sector_id"] = fundamentals["sector_id"].astype(str)
    fundamentals = fundamentals.sort_values(["sector_id", "publish_date"])

    pieces: list[pd.DataFrame] = []
    for sector_id, daily_group in daily.groupby("sector_id", sort=False):
        fund_group = fundamentals.loc[fundamentals["sector_id"] == sector_id].sort_values(
            "publish_date"
        )
        pieces.append(
            pd.merge_asof(
                daily_group.sort_values("date"),
                fund_group[["publish_date", "roe_ttm", "revenue_yoy", "profit_yoy"]],
                left_on="date",
                right_on="publish_date",
                direction="backward",
            )
        )
    asof = pd.concat(pieces, ignore_index=True)
    for column in ("roe_ttm", "revenue_yoy", "profit_yoy"):
        asof[f"{column}_z"] = _cross_sectional_z(asof, column)
    asof["financial"] = (
        0.4 * asof["roe_ttm_z"] + 0.35 * asof["revenue_yoy_z"] + 0.25 * asof["profit_yoy_z"]
    )
    return asof[["date", "sector_id", "financial"]]


def _flow(flows: pd.DataFrame, prices: pd.DataFrame, config: FactorConfig) -> pd.DataFrame:
    main = flows.pivot(index="date", columns="sector_id", values="main_net_inflow")
    north = flows.pivot(index="date", columns="sector_id", values="northbound_net")
    amount = prices.pivot(index="date", columns="sector_id", values="amount")
    window = config.flow_window
    main_sum = main.rolling(window, min_periods=window).sum()
    north_sum = north.rolling(window, min_periods=window).sum()
    amount_sum = amount.rolling(window, min_periods=window).sum().replace(0.0, np.nan)
    flow_raw = (main_sum + 0.5 * north_sum) / amount_sum
    long = flow_raw.stack().rename("flow_raw").reset_index()
    long["sector_id"] = long["sector_id"].astype(str)
    long["flow"] = _cross_sectional_z(long, "flow_raw")
    return long[["date", "sector_id", "flow"]]


def _crowding(flows: pd.DataFrame, config: FactorConfig) -> pd.DataFrame:
    crowd = flows.pivot(index="date", columns="sector_id", values="crowding_raw")
    crowd = crowd.rolling(config.crowding_window, min_periods=1).mean()
    long = crowd.stack().rename("crowding_raw").reset_index()
    long["sector_id"] = long["sector_id"].astype(str)
    long["crowding"] = _cross_sectional_z(long, "crowding_raw")
    return long[["date", "sector_id", "crowding"]]


def compute_factors(dataset: Dataset, config: FactorConfig | None = None) -> pd.DataFrame:
    """计算板块因子表，返回按 date, sector_id 排序的 DataFrame。"""

    config = config or FactorConfig()
    prices = dataset.prices.copy()
    prices["sector_id"] = prices["sector_id"].astype(str)
    prices = prices.sort_values(["date", "sector_id"])

    flows = dataset.flows.copy()
    flows["sector_id"] = flows["sector_id"].astype(str)
    flows = flows.sort_values(["date", "sector_id"])

    momentum = _momentum(prices, config)
    financial = _financial(dataset)
    flow = _flow(flows, prices, config)
    crowding = _crowding(flows, config)

    factors = momentum.merge(financial, on=["date", "sector_id"], how="left")
    factors = factors.merge(flow, on=["date", "sector_id"], how="left")
    factors = factors.merge(crowding, on=["date", "sector_id"], how="left")
    factors["crowding"] = factors["crowding"].fillna(0.0)
    factors["composite"] = (
        factors[["momentum", "financial", "flow"]].mean(axis=1)
        - config.crowding_weight * factors["crowding"]
    )
    factors = factors.dropna(subset=["momentum", "financial", "flow"])
    return factors.sort_values(["date", "sector_id"]).reset_index(drop=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_factors(
    factors: pd.DataFrame,
    out_dir: Path | str,
    config: FactorConfig | None = None,
) -> dict[str, Any]:
    """写入 factors.csv + manifest.json。"""

    config = config or FactorConfig()
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "factors.csv"
    factors.to_csv(
        path,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        float_format="%.6f",
        date_format="%Y-%m-%d",
    )
    manifest = {
        "schema_version": 1,
        "rows": int(len(factors)),
        "start": factors["date"].min().date().isoformat(),
        "end": factors["date"].max().date().isoformat(),
        "columns": list(factors.columns),
        "config": asdict(config),
        "sha256": _sha256_file(path),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def read_factors(path: Path | str) -> pd.DataFrame:
    """读取 factors.csv 或包含 factors.csv 的目录。"""

    path = Path(path)
    if path.is_dir():
        path = path / "factors.csv"
    if not path.is_file():
        raise DataSourceError(f"因子文件不存在：{path}")

    try:
        factors = pd.read_csv(path, parse_dates=["date"], dtype={"sector_id": "string"})
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise DataSourceError(f"因子文件读取失败：{exc}") from exc

    missing = [column for column in FACTOR_COLUMNS if column not in factors.columns]
    if missing:
        raise DataSourceError(f"因子文件缺少字段：{', '.join(missing)}")
    return factors.sort_values(["date", "sector_id"]).reset_index(drop=True)


__all__ = [
    "FACTOR_COLUMNS",
    "FactorConfig",
    "compute_factors",
    "read_factors",
    "write_factors",
]
