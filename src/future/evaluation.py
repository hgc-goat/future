"""回测、模型和漂移监控共用的评估工具。"""

from __future__ import annotations

import pandas as pd

from future.data.base import DataSourceError


def forward_returns(close_wide: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """t 日信号对应的前瞻收益：t+1 收盘到 t+h+1 收盘。"""

    if horizon <= 0:
        raise ValueError("horizon 必须为正数")
    return close_wide.shift(-(horizon + 1)) / close_wide.shift(-1) - 1.0


def target_end_dates(index: pd.DatetimeIndex, horizon: int) -> pd.Series:
    """每个信号日期对应目标收益的结束日期，用于训练集防泄漏。"""

    return pd.Series(index, index=index).shift(-(horizon + 1))


def daily_rank_ic(
    factors: pd.DataFrame,
    forward: pd.DataFrame,
    score_column: str,
    min_sectors: int = 5,
) -> pd.DataFrame:
    """逐日横截面 Rank IC。"""

    if score_column not in factors.columns:
        raise DataSourceError(f"因子表缺少 {score_column}")
    factor_wide = factors.pivot(index="date", columns="sector_id", values=score_column)
    rows: list[dict[str, object]] = []
    for current_date in factor_wide.index:
        if current_date not in forward.index:
            continue
        pair = pd.concat(
            [
                factor_wide.loc[current_date].rename("factor"),
                forward.loc[current_date].rename("forward"),
            ],
            axis=1,
        ).dropna()
        if len(pair) < min_sectors:
            continue
        ic = pair["factor"].rank().corr(pair["forward"].rank())
        if pd.notna(ic):
            rows.append({"date": current_date, "ic": float(ic)})
    return pd.DataFrame(rows, columns=["date", "ic"])


__all__ = ["daily_rank_ic", "forward_returns", "target_end_dates"]
