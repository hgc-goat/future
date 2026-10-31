"""新闻情感的同向确认与增量分析。

新闻不单独触发交易：只有同时满足
- news_count >= 3；
- |sentiment| >= 0.1；
- sentiment 与基础 composite 同向；
才把 ``boost * sentiment`` 叠加到 composite 上。否则 news_confirm = 0。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from future.backtest import BacktestConfig, run_backtest
from future.data.base import Dataset, DataSourceError
from future.factors import FACTOR_COLUMNS

FUSED_COLUMNS = (
    "date",
    "sector_id",
    "momentum_20",
    "momentum_60",
    "momentum_120",
    "momentum",
    "financial",
    "flow",
    "crowding",
    "composite_base",
    "sentiment",
    "news_count",
    "news_signal",
    "news_confirm",
    "composite_fused",
)


@dataclass(frozen=True)
class NewsConfig:
    """新闻确认阈值与叠加权重。"""

    min_news_count: int = 3
    min_abs_sentiment: float = 0.1
    boost: float = 0.15

    def __post_init__(self) -> None:
        if self.min_news_count < 0:
            raise ValueError("min_news_count 不能为负")
        if not 0.0 <= self.min_abs_sentiment <= 1.0:
            raise ValueError("min_abs_sentiment 必须在 [0, 1] 内")
        if self.boost < 0:
            raise ValueError("boost 不能为负")


def compute_news_fusion(
    factors: pd.DataFrame,
    news: pd.DataFrame,
    config: NewsConfig | None = None,
) -> pd.DataFrame:
    """生成 news_confirm、composite_base 与 composite_fused。"""

    config = config or NewsConfig()
    base = factors.copy()
    base["sector_id"] = base["sector_id"].astype(str)
    if "composite" not in base.columns:
        raise DataSourceError("因子表缺少 composite 字段")

    news_frame = news.copy()
    news_frame["sector_id"] = news_frame["sector_id"].astype(str)
    news_frame = news_frame[["date", "sector_id", "sentiment", "news_count"]]
    merged = base.merge(news_frame, on=["date", "sector_id"], how="left")
    merged["sentiment"] = merged["sentiment"].fillna(0.0)
    merged["news_count"] = merged["news_count"].fillna(0).astype(int)

    eligible = (merged["news_count"] >= config.min_news_count) & (
        merged["sentiment"].abs() >= config.min_abs_sentiment
    )
    merged["news_signal"] = np.where(eligible, merged["sentiment"], 0.0)
    same_direction = (np.sign(merged["news_signal"]) == np.sign(merged["composite"])) & (
        merged["news_signal"] != 0
    )
    merged["news_confirm"] = np.where(same_direction, merged["news_signal"], 0.0)
    merged["composite_base"] = merged["composite"]
    merged["composite_fused"] = merged["composite_base"] + config.boost * merged["news_confirm"]

    base_columns = [column for column in FACTOR_COLUMNS if column != "composite"]
    merged = merged[
        [
            *base_columns,
            "composite_base",
            "sentiment",
            "news_count",
            "news_signal",
            "news_confirm",
            "composite_fused",
        ]
    ]
    return merged.sort_values(["date", "sector_id"]).reset_index(drop=True)


def news_coverage(fused: pd.DataFrame) -> dict[str, Any]:
    """统计新闻覆盖率和同向确认比例。"""

    eligible = fused["news_signal"].abs() > 1e-12
    confirmed = fused["news_confirm"].abs() > 1e-12
    total = int(len(fused))
    eligible_count = int(eligible.sum())
    confirmed_count = int(confirmed.sum())
    same_sign_rate = None
    if eligible_count:
        same_sign_rate = float(
            (fused.loc[eligible, "news_signal"] * fused.loc[eligible, "composite_base"] > 0).mean()
        )
    return {
        "total_rows": total,
        "eligible_rows": eligible_count,
        "confirmed_rows": confirmed_count,
        "eligible_rate": float(eligible_count / total) if total else None,
        "confirmed_rate": float(confirmed_count / total) if total else None,
        "same_sign_rate_among_eligible": same_sign_rate,
        "avg_confirmed_sentiment": (
            float(fused.loc[confirmed, "news_confirm"].mean()) if confirmed_count else None
        ),
    }


def _corr_matrix(frame: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    columns = [
        "sentiment",
        "news_confirm",
        "momentum",
        "financial",
        "flow",
        "composite_base",
    ]
    matrix = frame[columns].corr()
    return {
        row: {
            column: (None if pd.isna(matrix.loc[row, column]) else float(matrix.loc[row, column]))
            for column in matrix.columns
        }
        for row in matrix.index
    }


def _spearman_matrix(frame: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    columns = [
        "sentiment",
        "news_confirm",
        "momentum",
        "financial",
        "flow",
        "composite_base",
    ]
    matrix = frame[columns].rank().corr()
    return {
        row: {
            column: (None if pd.isna(matrix.loc[row, column]) else float(matrix.loc[row, column]))
            for column in matrix.columns
        }
        for row in matrix.index
    }


def _delta(base: dict[str, Any], fused: dict[str, Any], key: str) -> float | None:
    if base.get(key) is None or fused.get(key) is None:
        return None
    return float(fused[key] - base[key])


def analyze_news_increment(
    dataset: Dataset,
    fused: pd.DataFrame,
    config: BacktestConfig | None = None,
    news_config: NewsConfig | None = None,
) -> dict[str, Any]:
    """比较基础 composite 与新闻确认后的 composite。"""

    config = config or BacktestConfig()
    news_config = news_config or NewsConfig()
    base_frame = fused.assign(composite=fused["composite_base"])
    fused_frame = fused.assign(composite=fused["composite_fused"])

    base_metrics = run_backtest(dataset, base_frame, config).metrics
    fused_metrics = run_backtest(dataset, fused_frame, config).metrics
    base_long_short = base_metrics["long_short"]
    fused_long_short = fused_metrics["long_short"]

    delta: dict[str, Any] = {}
    for horizon in ("5", "20"):
        base_ic = base_metrics["oos_ic"]["composite"][horizon]["mean_ic"]
        fused_ic = fused_metrics["oos_ic"]["composite"][horizon]["mean_ic"]
        delta[f"ic_{horizon}"] = (
            None if base_ic is None or fused_ic is None else float(fused_ic - base_ic)
        )
    for key in ("annualized_return", "sharpe", "max_drawdown", "win_rate"):
        delta[key] = _delta(base_long_short, fused_long_short, key)

    return {
        "news_config": asdict(news_config),
        "coverage": news_coverage(fused),
        "correlations": {
            "pearson": _corr_matrix(fused),
            "spearman": _spearman_matrix(fused),
        },
        "backtest": {
            "base": base_metrics,
            "fused": fused_metrics,
        },
        "delta": delta,
    }


def read_fused_factors(path: Path | str) -> pd.DataFrame:
    """读取 fused_factors.csv 或包含它的目录。"""

    path = Path(path)
    if path.is_dir():
        path = path / "fused_factors.csv"
    if not path.is_file():
        raise DataSourceError(f"融合因子文件不存在：{path}")
    try:
        factors = pd.read_csv(path, parse_dates=["date"], dtype={"sector_id": "string"})
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise DataSourceError(f"融合因子文件读取失败：{exc}") from exc
    missing = [column for column in FUSED_COLUMNS if column not in factors.columns]
    if missing:
        raise DataSourceError(f"融合因子文件缺少字段：{', '.join(missing)}")
    return factors.sort_values(["date", "sector_id"]).reset_index(drop=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_news_outputs(
    fused: pd.DataFrame,
    analysis: dict[str, Any],
    factor_dir: Path | str,
    analysis_dir: Path | str,
    news_config: NewsConfig | None = None,
) -> dict[str, Any]:
    """写入 fused_factors.csv 与 analysis.json。"""

    news_config = news_config or NewsConfig()
    factor_output = Path(factor_dir)
    factor_output.mkdir(parents=True, exist_ok=True)
    factor_path = factor_output / "fused_factors.csv"
    fused.to_csv(
        factor_path,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        float_format="%.6f",
        date_format="%Y-%m-%d",
    )
    factor_manifest = {
        "schema_version": 1,
        "rows": int(len(fused)),
        "start": fused["date"].min().date().isoformat(),
        "end": fused["date"].max().date().isoformat(),
        "config": asdict(news_config),
        "sha256": _sha256_file(factor_path),
    }
    (factor_output / "manifest.json").write_text(
        json.dumps(factor_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    analysis_output = Path(analysis_dir)
    analysis_output.mkdir(parents=True, exist_ok=True)
    analysis_path = analysis_output / "analysis.json"
    analysis_path.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    analysis_manifest = {
        "schema_version": 1,
        "sha256": _sha256_file(analysis_path),
        "coverage": analysis["coverage"],
        "delta": analysis["delta"],
    }
    (analysis_output / "manifest.json").write_text(
        json.dumps(analysis_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "factor_dir": str(factor_output),
        "analysis_dir": str(analysis_output),
        "factor_manifest": factor_manifest,
        "analysis_manifest": analysis_manifest,
    }


__all__ = [
    "FUSED_COLUMNS",
    "NewsConfig",
    "analyze_news_increment",
    "compute_news_fusion",
    "news_coverage",
    "read_fused_factors",
    "write_news_outputs",
]
