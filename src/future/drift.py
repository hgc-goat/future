"""漂移监控：滚动 IC 与因子分布 PSI，输出 0–1 严重度。"""

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
from future.evaluation import daily_rank_ic, forward_returns, target_end_dates

DRIFT_FEATURES = ("momentum", "financial", "flow", "crowding", "sentiment")


@dataclass(frozen=True)
class DriftConfig:
    """漂移监控窗口与参考区间。"""

    horizon: int = 20
    window: int = 20
    train_start: date = date(2021, 1, 1)
    train_end: date = date(2023, 12, 31)
    oos_start: date = date(2024, 1, 1)
    oos_end: date = date(2025, 12, 31)
    score_column: str = "composite_fused"
    psi_bins: int = 10
    psi_scale: float = 0.5
    min_sectors: int = 5

    def __post_init__(self) -> None:
        if not self.train_start < self.train_end < self.oos_start < self.oos_end:
            raise ValueError("要求 train_start < train_end < oos_start < oos_end")
        if self.horizon <= 0 or self.window < 2:
            raise ValueError("horizon 必须为正数，window 至少为 2")
        if self.psi_bins < 2 or self.psi_scale <= 0:
            raise ValueError("psi_bins >= 2 且 psi_scale > 0")


@dataclass
class DriftResult:
    metrics: dict[str, Any]
    rolling_ic: pd.DataFrame


def population_stability_index(
    expected: pd.Series | np.ndarray,
    actual: pd.Series | np.ndarray,
    bins: int = 10,
) -> float:
    """计算 PSI；完全相同分布约等于 0。"""

    exp = pd.Series(expected, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    act = pd.Series(actual, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    if exp.empty or act.empty:
        return 0.0
    edges = np.unique(np.quantile(exp.to_numpy(), np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    exp_counts = np.histogram(exp.to_numpy(), bins=edges)[0] / len(exp)
    act_counts = np.histogram(act.to_numpy(), bins=edges)[0] / len(act)
    eps = 1e-6
    exp_pct = np.clip(exp_counts, eps, None)
    act_pct = np.clip(act_counts, eps, None)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def _close_wide(dataset: Dataset) -> pd.DataFrame:
    prices = dataset.prices.copy()
    prices["sector_id"] = prices["sector_id"].astype(str)
    return prices.pivot(index="date", columns="sector_id", values="close").sort_index()


def _ic_with_target_end(
    dataset: Dataset,
    factors: pd.DataFrame,
    config: DriftConfig,
) -> pd.DataFrame:
    close_wide = _close_wide(dataset)
    forward = forward_returns(close_wide, config.horizon)
    ic_frame = daily_rank_ic(
        factors,
        forward,
        config.score_column,
        min_sectors=config.min_sectors,
    )
    end_map = target_end_dates(close_wide.index, config.horizon).rename("target_end")
    ic_frame["target_end"] = ic_frame["date"].map(end_map)
    return ic_frame


def _severity_level(value: float) -> str:
    if value < 0.3:
        return "low"
    if value < 0.6:
        return "medium"
    return "high"


def compute_drift(
    dataset: Dataset,
    factors: pd.DataFrame,
    config: DriftConfig | None = None,
) -> DriftResult:
    """计算滚动 IC、PSI 与综合漂移严重度。"""

    config = config or DriftConfig()
    if config.score_column not in factors.columns:
        raise DataSourceError(f"因子表缺少 {config.score_column}")
    missing = [column for column in DRIFT_FEATURES if column not in factors.columns]
    if missing:
        raise DataSourceError(f"漂移特征缺少字段：{', '.join(missing)}")

    factors = factors.copy()
    factors["sector_id"] = factors["sector_id"].astype(str)
    factors = factors.sort_values(["date", "sector_id"])

    ic_frame = _ic_with_target_end(dataset, factors, config)
    train_ic = ic_frame.loc[
        (ic_frame["date"] >= pd.Timestamp(config.train_start))
        & (ic_frame["date"] <= pd.Timestamp(config.train_end))
        & (ic_frame["target_end"] <= pd.Timestamp(config.train_end)),
        "ic",
    ].dropna()
    if train_ic.empty:
        raise DataSourceError("训练期没有可用的 IC 参考样本")

    train_mean = float(train_ic.mean())
    train_std = float(train_ic.std(ddof=1)) if len(train_ic) > 1 else 0.0

    oos_ic = ic_frame.loc[
        (ic_frame["date"] >= pd.Timestamp(config.oos_start))
        & (ic_frame["date"] <= pd.Timestamp(config.oos_end))
    ].copy()
    oos_ic = oos_ic.sort_values("date")
    min_periods = max(5, config.window // 2)
    oos_ic["rolling_ic"] = (
        oos_ic["ic"]
        .rolling(
            config.window,
            min_periods=min_periods,
        )
        .mean()
    )
    latest_rows = oos_ic.dropna(subset=["rolling_ic"]).tail(1)
    latest_ic = None if latest_rows.empty else float(latest_rows["rolling_ic"].iloc[0])
    latest_date = None if latest_rows.empty else latest_rows["date"].iloc[0].date().isoformat()

    if latest_ic is None:
        ic_severity = 0.0
    else:
        z_score = (train_mean - latest_ic) / max(train_std, 0.01)
        ic_severity = float(np.clip(z_score / 2.0, 0.0, 1.0))

    oos_dates = sorted(
        factors.loc[
            (factors["date"] >= pd.Timestamp(config.oos_start))
            & (factors["date"] <= pd.Timestamp(config.oos_end)),
            "date",
        ].unique()
    )
    current_dates = oos_dates[-config.window :]
    train_rows = factors.loc[
        (factors["date"] >= pd.Timestamp(config.train_start))
        & (factors["date"] <= pd.Timestamp(config.train_end))
    ]
    current_rows = factors.loc[factors["date"].isin(current_dates)]
    if current_rows.empty:
        raise DataSourceError("样本外窗口没有可用的漂移特征")

    psi_by_feature = {
        feature: population_stability_index(
            train_rows[feature],
            current_rows[feature],
            bins=config.psi_bins,
        )
        for feature in DRIFT_FEATURES
    }
    max_psi = max(psi_by_feature.values())
    average_psi = float(np.mean(list(psi_by_feature.values())))
    distribution_severity = float(np.clip(max_psi / config.psi_scale, 0.0, 1.0))
    overall = float(0.6 * ic_severity + 0.4 * distribution_severity)

    metrics: dict[str, Any] = {
        "config": {
            **asdict(config),
            "train_start": config.train_start.isoformat(),
            "train_end": config.train_end.isoformat(),
            "oos_start": config.oos_start.isoformat(),
            "oos_end": config.oos_end.isoformat(),
        },
        "train": {
            "ic_mean": train_mean,
            "ic_std": train_std,
            "n": int(len(train_ic)),
        },
        "latest": {
            "date": latest_date,
            "rolling_ic": latest_ic,
            "ic": (None if latest_rows.empty else float(latest_rows["ic"].iloc[0])),
        },
        "psi": {
            "by_feature": psi_by_feature,
            "max": max_psi,
            "average": average_psi,
        },
        "severity": {
            "ic": ic_severity,
            "distribution": distribution_severity,
            "overall": overall,
            "level": _severity_level(overall),
        },
    }
    rolling = oos_ic[["date", "ic", "rolling_ic"]].reset_index(drop=True)
    return DriftResult(metrics=metrics, rolling_ic=rolling)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_drift_outputs(result: DriftResult, out_dir: Path | str) -> dict[str, Any]:
    """写入 rolling_ic.csv、drift.json 和 manifest。"""

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    rolling_path = output / "rolling_ic.csv"
    result.rolling_ic.to_csv(
        rolling_path,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        float_format="%.6f",
        date_format="%Y-%m-%d",
    )
    drift_path = output / "drift.json"
    drift_path.write_text(
        json.dumps(result.metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "files": {
            "rolling_ic.csv": {
                "rows": int(len(result.rolling_ic)),
                "sha256": _sha256_file(rolling_path),
            },
            "drift.json": {"sha256": _sha256_file(drift_path)},
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


__all__ = [
    "DRIFT_FEATURES",
    "DriftConfig",
    "DriftResult",
    "compute_drift",
    "population_stability_index",
    "write_drift_outputs",
]
