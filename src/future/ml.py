"""HGB 进阶模型：未来 20 日横截面中位数标签。

- 训练期：2021-01-01 ~ 2023-12-31；为避免标签泄漏，训练样本的目标结束日
  必须不晚于训练期结束。
- 样本外：2024-01-01 ~ 2025-12-31。
- 以未来 20 日收益是否高于当日横截面中位数为二分类标签；
- 与阶段 3 的线性 composite_fused 基线做同口径回测对比。
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

from future.backtest import BacktestConfig, run_backtest
from future.data.base import Dataset, DataSourceError
from future.evaluation import forward_returns, target_end_dates

FEATURE_COLUMNS = (
    "momentum_20",
    "momentum_60",
    "momentum_120",
    "financial",
    "flow",
    "crowding",
    "sentiment",
    "news_count",
    "news_confirm",
)


class ModelUnavailableError(RuntimeError):
    """可选 ML 依赖未安装。"""


@dataclass(frozen=True)
class ModelConfig:
    """HGB 训练、切分与标签参数。"""

    horizon: int = 20
    train_start: date = date(2021, 1, 1)
    train_end: date = date(2023, 12, 31)
    oos_start: date = date(2024, 1, 1)
    oos_end: date = date(2025, 12, 31)
    random_state: int = 42
    max_iter: int = 150
    learning_rate: float = 0.05
    max_leaf_nodes: int = 15
    min_samples_leaf: int = 50
    l2_regularization: float = 0.1

    def __post_init__(self) -> None:
        if not self.train_start < self.train_end < self.oos_start < self.oos_end:
            raise ValueError("要求 train_start < train_end < oos_start < oos_end")
        if self.horizon <= 0:
            raise ValueError("horizon 必须为正数")
        if self.max_iter <= 0 or self.max_leaf_nodes < 2:
            raise ValueError("HGB 参数不合法")


@dataclass
class AdvancedResult:
    metrics: dict[str, Any]
    predictions: pd.DataFrame


def _close_wide(dataset: Dataset) -> pd.DataFrame:
    prices = dataset.prices.copy()
    prices["sector_id"] = prices["sector_id"].astype(str)
    return prices.pivot(index="date", columns="sector_id", values="close").sort_index()


def _labels(close_wide: pd.DataFrame, horizon: int) -> pd.DataFrame:
    forward = forward_returns(close_wide, horizon)
    median = forward.median(axis=1)
    labels = forward.gt(median, axis=0).astype(float).mask(forward.isna())
    long = labels.stack().rename("label").reset_index()
    end_map = target_end_dates(close_wide.index, horizon)
    long["target_end"] = long["date"].map(end_map)
    return long


def _feature_frame(factors: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in FEATURE_COLUMNS if column not in factors.columns]
    if missing:
        raise DataSourceError(f"特征缺少字段：{', '.join(missing)}")
    if "composite_fused" not in factors.columns:
        raise DataSourceError("进阶模型需要 composite_fused 作为线性基线")

    features = factors.copy()
    features["sector_id"] = features["sector_id"].astype(str)
    for column in FEATURE_COLUMNS:
        features[column] = pd.to_numeric(features[column], errors="coerce").fillna(0.0)
    return features.sort_values(["date", "sector_id"]).reset_index(drop=True)


def _config_payload(config: ModelConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key in ("train_start", "train_end", "oos_start", "oos_end"):
        payload[key] = getattr(config, key).isoformat()
    return payload


def _delta(base: dict[str, Any], hgb: dict[str, Any]) -> dict[str, float | None]:
    base_ic = base["oos_ic"]["composite"]
    hgb_ic = hgb["oos_ic"]["composite"]
    delta: dict[str, float | None] = {}
    for horizon in ("5", "20"):
        base_value = base_ic[horizon]["mean_ic"]
        hgb_value = hgb_ic[horizon]["mean_ic"]
        delta[f"ic_{horizon}"] = (
            None if base_value is None or hgb_value is None else float(hgb_value - base_value)
        )
    for key in ("annualized_return", "sharpe", "max_drawdown", "win_rate"):
        base_value = base["long_short"][key]
        hgb_value = hgb["long_short"][key]
        delta[key] = (
            None if base_value is None or hgb_value is None else float(hgb_value - base_value)
        )
    return delta


def train_advanced_model(
    dataset: Dataset,
    factors: pd.DataFrame,
    config: ModelConfig | None = None,
) -> AdvancedResult:
    """训练 HGB 并对比线性融合基线。"""

    config = config or ModelConfig()
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
    except ImportError as exc:
        raise ModelUnavailableError(
            "未安装 scikit-learn；请执行 uv pip install -e '.[ml]' 后重试"
        ) from exc

    features = _feature_frame(factors)
    close_wide = _close_wide(dataset)
    labels = _labels(close_wide, config.horizon)
    merged = features.merge(labels, on=["date", "sector_id"], how="inner")
    train = merged.loc[
        (merged["date"] >= pd.Timestamp(config.train_start))
        & (merged["date"] <= pd.Timestamp(config.train_end))
        & (merged["target_end"] <= pd.Timestamp(config.train_end))
        & merged["label"].notna()
    ].copy()
    if len(train) < 100:
        raise DataSourceError("训练样本不足，至少需要 100 条带标签样本")

    y_train = train["label"].astype(int).to_numpy()
    if len(np.unique(y_train)) < 2:
        raise DataSourceError("训练标签只有一个类别，无法训练分类器")

    x_train = train[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    model = HistGradientBoostingClassifier(
        max_iter=config.max_iter,
        learning_rate=config.learning_rate,
        max_leaf_nodes=config.max_leaf_nodes,
        min_samples_leaf=config.min_samples_leaf,
        l2_regularization=config.l2_regularization,
        early_stopping=False,
        random_state=config.random_state,
    )
    model.fit(x_train, y_train)

    x_all = features[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    classes = model.classes_
    positive_index = int(np.where(classes == 1)[0][0])
    probabilities = model.predict_proba(x_all)[:, positive_index]
    predictions = features[["date", "sector_id"]].copy()
    predictions["p_up"] = probabilities

    oos_mask = (features["date"] >= pd.Timestamp(config.oos_start)) & (
        features["date"] <= pd.Timestamp(config.oos_end)
    )
    oos_rows = int(oos_mask.sum())
    backtest_config = BacktestConfig(
        train_start=config.train_start,
        train_end=config.train_end,
        oos_start=config.oos_start,
        oos_end=config.oos_end,
    )
    baseline_frame = features.assign(composite=features["composite_fused"])
    hgb_frame = features.merge(predictions, on=["date", "sector_id"], how="left")
    hgb_frame["composite"] = hgb_frame["p_up"]
    baseline_metrics = run_backtest(dataset, baseline_frame, backtest_config).metrics
    hgb_metrics = run_backtest(dataset, hgb_frame, backtest_config).metrics

    metrics: dict[str, Any] = {
        "model": {
            "class": "HistGradientBoostingClassifier",
            "config": _config_payload(config),
            "features": list(FEATURE_COLUMNS),
        },
        "split": {
            "train_start": config.train_start.isoformat(),
            "train_end": config.train_end.isoformat(),
            "oos_start": config.oos_start.isoformat(),
            "oos_end": config.oos_end.isoformat(),
        },
        "train_rows": int(len(train)),
        "oos_rows": oos_rows,
        "train_label_end": train["target_end"].max().date().isoformat(),
        "baseline": baseline_metrics,
        "hgb": hgb_metrics,
        "delta": _delta(baseline_metrics, hgb_metrics),
    }
    return AdvancedResult(metrics=metrics, predictions=predictions)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_advanced_outputs(result: AdvancedResult, out_dir: Path | str) -> dict[str, Any]:
    """写入 predictions.csv、metrics.json 和 manifest。"""

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions_path = output / "predictions.csv"
    result.predictions.to_csv(
        predictions_path,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        float_format="%.6f",
        date_format="%Y-%m-%d",
    )
    metrics_path = output / "metrics.json"
    metrics_path.write_text(
        json.dumps(result.metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "files": {
            "predictions.csv": {
                "rows": int(len(result.predictions)),
                "sha256": _sha256_file(predictions_path),
            },
            "metrics.json": {"sha256": _sha256_file(metrics_path)},
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


__all__ = [
    "FEATURE_COLUMNS",
    "AdvancedResult",
    "ModelConfig",
    "ModelUnavailableError",
    "train_advanced_model",
    "write_advanced_outputs",
]
