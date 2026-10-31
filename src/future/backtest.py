"""月度分层回测与样本外 IC。

- 训练期：2021-01-01 ~ 2023-12-31（仅用于标记切分，本阶段基线不做参数拟合）；
- 样本外：2024-01-01 ~ 2025-12-31；
- 每月最后一个交易日生成信号，下一交易日开始持有，避免前视；
- 5 层等权组合 + 顶底多空，报告收益、波动、Sharpe、最大回撤与 OOS IC。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from future.data.base import Dataset, DataSourceError
from future.factors import FACTOR_COLUMNS

IC_FACTORS = ("momentum", "financial", "flow", "crowding", "composite")


@dataclass(frozen=True)
class BacktestConfig:
    """回测切分与分层参数。"""

    train_start: date = date(2021, 1, 1)
    train_end: date = date(2023, 12, 31)
    oos_start: date = date(2024, 1, 1)
    oos_end: date = date(2025, 12, 31)
    n_layers: int = 5
    ic_horizons: tuple[int, ...] = (5, 20)
    min_sectors: int = 5

    def __post_init__(self) -> None:
        if not self.train_start < self.train_end < self.oos_start < self.oos_end:
            raise ValueError("要求 train_start < train_end < oos_start < oos_end")
        if self.n_layers < 2:
            raise ValueError("n_layers 至少为 2")
        if self.min_sectors < self.n_layers:
            raise ValueError("min_sectors 不能小于 n_layers")
        if not self.ic_horizons or any(horizon <= 0 for horizon in self.ic_horizons):
            raise ValueError("ic_horizons 必须是正数列表")


@dataclass
class BacktestResult:
    metrics: dict[str, Any]
    layer_returns: pd.DataFrame
    ic_series: pd.DataFrame


def _sector_returns(dataset: Dataset) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = dataset.prices.copy()
    prices["sector_id"] = prices["sector_id"].astype(str)
    prices = prices.sort_values(["sector_id", "date"])
    prices["return"] = prices.groupby("sector_id")["close"].pct_change(fill_method=None)
    returns_wide = prices.pivot(index="date", columns="sector_id", values="return").sort_index()
    close_wide = prices.pivot(index="date", columns="sector_id", values="close").sort_index()
    return returns_wide, close_wide


def _month_end_signals(
    trading_days: pd.DatetimeIndex, config: BacktestConfig
) -> list[pd.Timestamp]:
    if len(trading_days) == 0:
        return []
    month_ends = (
        pd.Series(trading_days, index=trading_days).groupby(trading_days.to_period("M")).max()
    )
    earliest = pd.Timestamp(config.oos_start) - pd.Timedelta(days=31)
    return [
        pd.Timestamp(value)
        for value in month_ends
        if earliest <= pd.Timestamp(value) <= pd.Timestamp(config.oos_end)
    ]


def _layer_weights(
    factors: pd.DataFrame,
    signal_dates: list[pd.Timestamp],
    config: BacktestConfig,
) -> dict[int, pd.DataFrame]:
    factor_wide = factors.pivot(index="date", columns="sector_id", values="composite")
    sectors = list(factor_wide.columns)
    weights = {
        layer: pd.DataFrame(0.0, index=signal_dates, columns=sectors)
        for layer in range(1, config.n_layers + 1)
    }

    for signal_date in signal_dates:
        if signal_date not in factor_wide.index:
            continue
        scores = factor_wide.loc[signal_date].dropna()
        if len(scores) < max(config.min_sectors, config.n_layers):
            continue
        ranks = scores.rank(method="first")
        labels = pd.qcut(ranks, config.n_layers, labels=False) + 1
        for layer in range(1, config.n_layers + 1):
            members = labels.index[labels == layer]
            if len(members):
                weights[layer].loc[signal_date, members] = 1.0 / len(members)
    return weights


def _effective_weights(weights: pd.DataFrame, returns_wide: pd.DataFrame) -> pd.DataFrame:
    return (
        weights.reindex(index=returns_wide.index, columns=returns_wide.columns)
        .ffill()
        .shift(1)
        .fillna(0.0)
    )


def _portfolio_metrics(returns: pd.Series) -> dict[str, Any]:
    clean = returns.dropna()
    if clean.empty:
        return {
            "cumulative_return": None,
            "annualized_return": None,
            "annualized_volatility": None,
            "sharpe": None,
            "max_drawdown": None,
            "win_rate": None,
            "n_days": 0,
        }

    equity = (1.0 + clean).cumprod()
    n_days = int(len(clean))
    cumulative = float(equity.iloc[-1] - 1.0)
    annualized = float(equity.iloc[-1] ** (252.0 / n_days) - 1.0) if equity.iloc[-1] > 0 else -1.0
    std = float(clean.std(ddof=1)) if n_days > 1 else 0.0
    volatility = std * np.sqrt(252.0)
    sharpe = float(clean.mean() / std * np.sqrt(252.0)) if std > 0 else 0.0
    drawdown = equity / equity.cummax() - 1.0
    return {
        "cumulative_return": cumulative,
        "annualized_return": annualized,
        "annualized_volatility": float(volatility),
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "win_rate": float((clean > 0).mean()),
        "n_days": n_days,
    }


def _ic_stats(values: pd.Series) -> dict[str, Any]:
    clean = values.dropna()
    if clean.empty:
        return {"mean_ic": None, "std_ic": None, "icir": None, "positive_ratio": None, "n": 0}
    mean = float(clean.mean())
    std = float(clean.std(ddof=1)) if len(clean) > 1 else 0.0
    return {
        "mean_ic": mean,
        "std_ic": std,
        "icir": float(mean / std) if std > 0 else None,
        "positive_ratio": float((clean > 0).mean()),
        "n": int(len(clean)),
    }


def _compute_ic(
    factors: pd.DataFrame,
    close_wide: pd.DataFrame,
    config: BacktestConfig,
) -> tuple[pd.DataFrame, dict[str, dict[str, dict[str, Any]]]]:
    rows: list[dict[str, Any]] = []
    for horizon in config.ic_horizons:
        forward = close_wide.shift(-(horizon + 1)) / close_wide.shift(-1) - 1.0
        for factor_name in IC_FACTORS:
            factor_wide = factors.pivot(
                index="date",
                columns="sector_id",
                values=factor_name,
            )
            dates = factor_wide.index[
                (factor_wide.index >= pd.Timestamp(config.oos_start))
                & (factor_wide.index <= pd.Timestamp(config.oos_end))
            ]
            for current_date in dates:
                if current_date not in forward.index:
                    continue
                pair = pd.concat(
                    [
                        factor_wide.loc[current_date].rename("factor"),
                        forward.loc[current_date].rename("forward"),
                    ],
                    axis=1,
                ).dropna()
                if len(pair) < config.min_sectors:
                    continue
                ic = pair["factor"].rank().corr(pair["forward"].rank())
                if pd.notna(ic):
                    rows.append(
                        {
                            "date": current_date,
                            "factor": factor_name,
                            "horizon": horizon,
                            "ic": float(ic),
                        }
                    )

    ic_frame = pd.DataFrame(rows, columns=["date", "factor", "horizon", "ic"])
    summary: dict[str, dict[str, dict[str, Any]]] = {}
    for factor_name in IC_FACTORS:
        summary[factor_name] = {}
        for horizon in config.ic_horizons:
            values = ic_frame.loc[
                (ic_frame["factor"] == factor_name) & (ic_frame["horizon"] == horizon),
                "ic",
            ]
            summary[factor_name][str(horizon)] = _ic_stats(values)
    return ic_frame, summary


def _config_payload(config: BacktestConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key in ("train_start", "train_end", "oos_start", "oos_end"):
        payload[key] = getattr(config, key).isoformat()
    payload["ic_horizons"] = list(config.ic_horizons)
    return payload


def run_backtest(
    dataset: Dataset,
    factors: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """运行月度分层回测，返回可写入文件的指标与序列。"""

    config = config or BacktestConfig()
    missing = [column for column in FACTOR_COLUMNS if column not in factors.columns]
    if missing:
        raise DataSourceError(f"因子表缺少字段：{', '.join(missing)}")

    factors = factors.copy()
    factors["sector_id"] = factors["sector_id"].astype(str)
    factors = factors.sort_values(["date", "sector_id"])

    returns_wide, close_wide = _sector_returns(dataset)
    signal_dates = _month_end_signals(returns_wide.index, config)
    if not signal_dates:
        raise DataSourceError("样本外区间没有可用的月末调仓日")

    weights = _layer_weights(factors, signal_dates, config)
    layer_returns = pd.DataFrame(index=returns_wide.index)
    for layer in range(1, config.n_layers + 1):
        daily_weights = _effective_weights(weights[layer], returns_wide)
        layer_returns[f"layer_{layer}"] = (daily_weights * returns_wide).sum(axis=1)
    layer_returns["long_short"] = (
        layer_returns[f"layer_{config.n_layers}"] - layer_returns["layer_1"]
    )

    first_signal = min(signal_dates)
    future_dates = returns_wide.index[returns_wide.index > first_signal]
    effective_start = pd.Timestamp(config.oos_start)
    if len(future_dates):
        effective_start = max(effective_start, pd.Timestamp(future_dates[0]))
    mask = (layer_returns.index >= effective_start) & (
        layer_returns.index <= pd.Timestamp(config.oos_end)
    )
    layer_returns = layer_returns.loc[mask].copy()
    if layer_returns.empty:
        raise DataSourceError("样本外区间没有有效的组合收益")

    ic_series, oos_ic = _compute_ic(factors, close_wide, config)
    layers_metrics = {
        str(layer): _portfolio_metrics(layer_returns[f"layer_{layer}"])
        for layer in range(1, config.n_layers + 1)
    }
    metrics: dict[str, Any] = {
        "config": _config_payload(config),
        "split": {
            "train_start": config.train_start.isoformat(),
            "train_end": config.train_end.isoformat(),
            "oos_start": config.oos_start.isoformat(),
            "oos_end": config.oos_end.isoformat(),
        },
        "n_rebalances": len(signal_dates),
        "effective_start": layer_returns.index.min().date().isoformat(),
        "effective_end": layer_returns.index.max().date().isoformat(),
        "n_dates": int(len(layer_returns)),
        "layers": layers_metrics,
        "long_short": _portfolio_metrics(layer_returns["long_short"]),
        "oos_ic": oos_ic,
    }
    layer_returns.index.name = "date"
    return BacktestResult(metrics=metrics, layer_returns=layer_returns, ic_series=ic_series)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_backtest(result: BacktestResult, out_dir: Path | str) -> dict[str, Any]:
    """写入指标、分层收益、IC 序列和 manifest。"""

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)

    layer_path = output / "layer_returns.csv"
    result.layer_returns.reset_index().to_csv(
        layer_path,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        float_format="%.6f",
        date_format="%Y-%m-%d",
    )
    ic_path = output / "ic_series.csv"
    result.ic_series.to_csv(
        ic_path,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        float_format="%.6f",
        date_format="%Y-%m-%d",
    )

    (output / "metrics.json").write_text(
        json.dumps(result.metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "files": {
            "layer_returns.csv": {
                "rows": int(len(result.layer_returns)),
                "sha256": _sha256_file(layer_path),
            },
            "ic_series.csv": {
                "rows": int(len(result.ic_series)),
                "sha256": _sha256_file(ic_path),
            },
            "metrics.json": {
                "sha256": _sha256_file(output / "metrics.json"),
            },
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "run_backtest",
    "write_backtest",
]
