"""命令行入口：生成、校验 Mock 数据，查看运行配置。"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd

from future import __version__
from future.backtest import BacktestConfig, run_backtest, write_backtest
from future.config import Settings
from future.data.base import DataSourceError
from future.data.mock import MockDataAdapter
from future.data.registry import resolve_adapter
from future.data.storage import CsvDataAdapter, read_dataset, verify_manifest, write_dataset
from future.data.validate import ValidationReport, validate_dataset
from future.drift import DriftConfig, compute_drift, write_drift_outputs
from future.factors import (
    FACTOR_COLUMNS,
    FactorConfig,
    compute_factors,
    read_factors,
    write_factors,
)
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
    read_fused_factors,
    write_news_outputs,
)
from future.pipeline import PipelineConfig, run_pipeline


def _date_arg(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("日期格式应为 YYYY-MM-DD") from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="future",
        description="板块趋势概率评估系统 CLI",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate-data", help="生成可复现 Mock 数据")
    generate.add_argument("--seed", type=int, default=None, help="随机种子，默认 42")
    generate.add_argument("--start", type=_date_arg, default=None, help="开始日期 YYYY-MM-DD")
    generate.add_argument("--end", type=_date_arg, default=None, help="结束日期 YYYY-MM-DD")
    generate.add_argument("--out", type=Path, default=None, help="输出目录，默认 data/mock")
    generate.add_argument("--json", action="store_true", help="输出 JSON 摘要")

    validate = subparsers.add_parser("validate-data", help="校验本地 CSV 数据")
    validate.add_argument("--path", type=Path, required=True, help="数据目录")
    validate.add_argument("--json", action="store_true", help="输出 JSON 结果")

    info = subparsers.add_parser("info", help="查看版本、数据源与环境")
    info.add_argument("--json", action="store_true", help="输出 JSON 结果")

    factors = subparsers.add_parser("compute-factors", help="计算结构化板块因子")
    factors.add_argument("--data-dir", type=Path, default=Path("data/mock"))
    factors.add_argument("--out", type=Path, default=Path("data/factors"))
    factors.add_argument("--start", type=_date_arg, default=date(2021, 1, 1))
    factors.add_argument("--end", type=_date_arg, default=date(2025, 12, 31))
    factors.add_argument("--json", action="store_true", help="输出 JSON 结果")

    backtest = subparsers.add_parser("backtest", help="运行样本外分层回测")
    backtest.add_argument("--data-dir", type=Path, default=Path("data/mock"))
    backtest.add_argument(
        "--factors",
        type=Path,
        default=Path("data/factors/factors.csv"),
    )
    backtest.add_argument("--out", type=Path, default=Path("artifacts/backtest"))
    backtest.add_argument("--oos-start", type=_date_arg, default=date(2024, 1, 1))
    backtest.add_argument("--oos-end", type=_date_arg, default=date(2025, 12, 31))
    backtest.add_argument("--layers", type=int, default=5)
    backtest.add_argument("--json", action="store_true", help="输出 JSON 结果")

    news = subparsers.add_parser("fuse-news", help="新闻情感同向确认与增量分析")
    news.add_argument("--data-dir", type=Path, default=Path("data/mock"))
    news.add_argument("--factors", type=Path, default=Path("data/factors/factors.csv"))
    news.add_argument("--out", type=Path, default=Path("data/factors_news"))
    news.add_argument("--analysis-out", type=Path, default=Path("artifacts/news"))
    news.add_argument("--oos-start", type=_date_arg, default=date(2024, 1, 1))
    news.add_argument("--oos-end", type=_date_arg, default=date(2025, 12, 31))
    news.add_argument("--layers", type=int, default=5)
    news.add_argument("--min-news-count", type=int, default=3)
    news.add_argument("--min-abs-sentiment", type=float, default=0.1)
    news.add_argument("--boost", type=float, default=0.15)
    news.add_argument("--json", action="store_true", help="输出 JSON 结果")

    advanced = subparsers.add_parser("advanced-model", help="HGB 进阶模型与线性基线对比")
    advanced.add_argument("--data-dir", type=Path, default=Path("data/mock"))
    advanced.add_argument(
        "--factors",
        type=Path,
        default=Path("data/factors_news/fused_factors.csv"),
    )
    advanced.add_argument("--out", type=Path, default=Path("artifacts/advanced"))
    advanced.add_argument("--train-start", type=_date_arg, default=date(2021, 1, 1))
    advanced.add_argument("--train-end", type=_date_arg, default=date(2023, 12, 31))
    advanced.add_argument("--oos-start", type=_date_arg, default=date(2024, 1, 1))
    advanced.add_argument("--oos-end", type=_date_arg, default=date(2025, 12, 31))
    advanced.add_argument("--horizon", type=int, default=20)
    advanced.add_argument("--json", action="store_true", help="输出 JSON 结果")

    drift = subparsers.add_parser("drift-monitor", help="滚动 IC 与分布漂移监控")
    drift.add_argument("--data-dir", type=Path, default=Path("data/mock"))
    drift.add_argument(
        "--factors",
        type=Path,
        default=Path("data/factors_news/fused_factors.csv"),
    )
    drift.add_argument("--out", type=Path, default=Path("artifacts/drift"))
    drift.add_argument("--train-start", type=_date_arg, default=date(2021, 1, 1))
    drift.add_argument("--train-end", type=_date_arg, default=date(2023, 12, 31))
    drift.add_argument("--oos-start", type=_date_arg, default=date(2024, 1, 1))
    drift.add_argument("--oos-end", type=_date_arg, default=date(2025, 12, 31))
    drift.add_argument("--horizon", type=int, default=20)
    drift.add_argument("--window", type=int, default=20)
    drift.add_argument("--score-column", default="composite_fused")
    drift.add_argument("--json", action="store_true", help="输出 JSON 结果")

    pipeline = subparsers.add_parser("pipeline", help="一键运行阶段 1–4 全部流水线")
    pipeline.add_argument("--root", type=Path, default=Path())
    pipeline.add_argument("--seed", type=int, default=42)
    pipeline.add_argument("--start", type=_date_arg, default=date(2020, 1, 2))
    pipeline.add_argument("--end", type=_date_arg, default=date(2025, 12, 31))
    pipeline.add_argument("--train-start", type=_date_arg, default=date(2021, 1, 1))
    pipeline.add_argument("--train-end", type=_date_arg, default=date(2023, 12, 31))
    pipeline.add_argument("--oos-start", type=_date_arg, default=date(2024, 1, 1))
    pipeline.add_argument("--oos-end", type=_date_arg, default=date(2025, 12, 31))
    pipeline.add_argument("--factor-start", type=_date_arg, default=date(2021, 1, 1))
    pipeline.add_argument("--sector-limit", type=int, default=None)
    pipeline.add_argument("--json", action="store_true", help="输出 JSON 摘要")

    web = subparsers.add_parser("web", help="启动 Dash 展示页面")
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=8050)
    web.add_argument("--debug", action="store_true")
    web.add_argument("--data-root", type=Path, default=Path())

    return parser


def _generate_data(args: argparse.Namespace, settings: Settings) -> int:
    seed = settings.seed if args.seed is None else args.seed
    start = settings.start if args.start is None else args.start
    end = settings.end if args.end is None else args.end
    out_dir = settings.data_dir if args.out is None else args.out

    try:
        adapter = MockDataAdapter(seed=seed, start=start, end=end)
        dataset = adapter.load(start, end)
        manifest = write_dataset(dataset, out_dir)
    except (ValueError, DataSourceError, OSError) as exc:
        print(f"生成失败：{exc}", file=sys.stderr)
        return 1

    rows = {name: int(entry["rows"]) for name, entry in manifest["files"].items()}
    payload = {
        "ok": True,
        "out": str(out_dir),
        "seed": seed,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "files": rows,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"已生成 Mock 数据：{out_dir}")
        print(f"seed={seed}，区间={start.isoformat()} ~ {end.isoformat()}")
        for filename, row_count in rows.items():
            print(f"  {filename}: {row_count} 行")
        print(f"  交易日: {manifest['n_days']}，板块: {manifest['n_sectors']}")
    return 0


def _validate_data(args: argparse.Namespace) -> int:
    report = ValidationReport()
    try:
        dataset = read_dataset(args.path)
        report = validate_dataset(dataset)
        report.errors.extend(verify_manifest(args.path))
    except DataSourceError as exc:
        report.errors.append(str(exc))

    payload = {
        "ok": report.ok,
        "path": str(args.path),
        "errors": report.errors,
        "warnings": report.warnings,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif report.ok:
        print(f"校验通过：{args.path}")
        for warning in report.warnings:
            print(f"  警告: {warning}")
    else:
        print(f"校验失败：{args.path}", file=sys.stderr)
        for error in report.errors:
            print(f"  - {error}", file=sys.stderr)
    return 0 if report.ok else 1


def _info(args: argparse.Namespace, settings: Settings) -> int:
    resolution = resolve_adapter(
        settings.data_source,
        data_dir=settings.data_dir,
        seed=settings.seed,
        start=settings.start,
        end=settings.end,
    )
    payload = {
        "version": __version__,
        "python": platform.python_version(),
        "seed": settings.seed,
        "start": settings.start.isoformat(),
        "end": settings.end.isoformat(),
        "data_source": settings.data_source,
        "data_dir": str(settings.data_dir),
        "adapter": type(resolution.adapter).__name__,
        "fallback_planned": resolution.fallback_planned,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"future {payload['version']} / Python {payload['python']}")
        print(f"数据源: {payload['data_source']} ({payload['adapter']})")
        print(f"数据目录: {payload['data_dir']}")
        print(f"区间: {payload['start']} ~ {payload['end']}，seed={payload['seed']}")
        if payload["fallback_planned"]:
            print("真实数据源失败时将自动降级到 Mock")
    return 0


def _compute_factors(args: argparse.Namespace) -> int:
    try:
        dataset = CsvDataAdapter(args.data_dir).load()
        config = FactorConfig()
        factors = compute_factors(dataset, config)
        factors = factors.loc[
            (factors["date"] >= pd.Timestamp(args.start))
            & (factors["date"] <= pd.Timestamp(args.end))
        ]
        if factors.empty:
            raise DataSourceError("所选日期区间没有有效因子")
        manifest = write_factors(factors, args.out, config)
    except (DataSourceError, ValueError, OSError) as exc:
        print(f"因子计算失败：{exc}", file=sys.stderr)
        return 1

    payload = {
        "ok": True,
        "out": str(args.out),
        "rows": int(len(factors)),
        "start": manifest["start"],
        "end": manifest["end"],
        "factor_columns": [
            column for column in FACTOR_COLUMNS if column not in ("date", "sector_id")
        ],
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"已计算因子：{args.out}")
        print(f"行数: {payload['rows']}，区间: {payload['start']} ~ {payload['end']}")
        print(f"因子: {', '.join(payload['factor_columns'])}")
    return 0


def _run_backtest(args: argparse.Namespace) -> int:
    try:
        dataset = CsvDataAdapter(args.data_dir).load()
        factors = read_factors(args.factors)
        config = BacktestConfig(
            oos_start=args.oos_start,
            oos_end=args.oos_end,
            n_layers=args.layers,
        )
        result = run_backtest(dataset, factors, config)
        write_backtest(result, args.out)
    except (DataSourceError, ValueError, OSError) as exc:
        print(f"回测失败：{exc}", file=sys.stderr)
        return 1

    long_short = result.metrics["long_short"]
    payload = {
        "ok": True,
        "out": str(args.out),
        "effective_start": result.metrics["effective_start"],
        "effective_end": result.metrics["effective_end"],
        "n_dates": result.metrics["n_dates"],
        "n_rebalances": result.metrics["n_rebalances"],
        "long_short": long_short,
        "oos_ic": result.metrics["oos_ic"],
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"回测完成：{args.out}")
        print(
            "样本外区间: "
            f"{payload['effective_start']} ~ {payload['effective_end']}，"
            f"{payload['n_dates']} 个交易日，{payload['n_rebalances']} 次调仓"
        )
        print(
            "多空组合: "
            f"年化 {long_short['annualized_return']:.2%}，"
            f"Sharpe {long_short['sharpe']:.2f}，"
            f"最大回撤 {long_short['max_drawdown']:.2%}"
        )
        for horizon, stats in result.metrics["oos_ic"]["composite"].items():
            if stats["mean_ic"] is not None:
                print(f"复合因子 {horizon} 日 OOS IC: {stats['mean_ic']:.4f}")
    return 0


def _fuse_news(args: argparse.Namespace) -> int:
    try:
        dataset = CsvDataAdapter(args.data_dir).load()
        factors = read_factors(args.factors)
        news_config = NewsConfig(
            min_news_count=args.min_news_count,
            min_abs_sentiment=args.min_abs_sentiment,
            boost=args.boost,
        )
        config = BacktestConfig(
            oos_start=args.oos_start,
            oos_end=args.oos_end,
            n_layers=args.layers,
        )
        fused = compute_news_fusion(factors, dataset.news, news_config)
        analysis = analyze_news_increment(dataset, fused, config, news_config)
        write_news_outputs(fused, analysis, args.out, args.analysis_out, news_config)
    except (DataSourceError, ValueError, OSError) as exc:
        print(f"新闻融合失败：{exc}", file=sys.stderr)
        return 1

    base = analysis["backtest"]["base"]
    fused_metrics = analysis["backtest"]["fused"]
    payload = {
        "ok": True,
        "out": str(args.out),
        "analysis_out": str(args.analysis_out),
        "rows": int(len(fused)),
        "coverage": analysis["coverage"],
        "delta": analysis["delta"],
        "base_ic20": base["oos_ic"]["composite"]["20"]["mean_ic"],
        "fused_ic20": fused_metrics["oos_ic"]["composite"]["20"]["mean_ic"],
        "base_long_short": base["long_short"],
        "fused_long_short": fused_metrics["long_short"],
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        coverage = payload["coverage"]
        print(f"新闻融合完成：{args.out}")
        print(
            "新闻覆盖: "
            f"可确认 {coverage['eligible_rate']:.2%}，"
            f"同向确认 {coverage['confirmed_rate']:.2%}"
        )
        print(f"复合因子 20 日 OOS IC: {payload['base_ic20']:.4f} -> {payload['fused_ic20']:.4f}")
        base_ls = payload["base_long_short"]
        fused_ls = payload["fused_long_short"]
        print(
            f"多空年化: {base_ls['annualized_return']:.2%} -> {fused_ls['annualized_return']:.2%}"
        )
        print(f"多空 Sharpe: {base_ls['sharpe']:.2f} -> {fused_ls['sharpe']:.2f}")
        print(f"多空最大回撤: {base_ls['max_drawdown']:.2%} -> {fused_ls['max_drawdown']:.2%}")
    return 0


def _advanced_model(args: argparse.Namespace) -> int:
    try:
        dataset = CsvDataAdapter(args.data_dir).load()
        factors = read_fused_factors(args.factors)
        config = ModelConfig(
            horizon=args.horizon,
            train_start=args.train_start,
            train_end=args.train_end,
            oos_start=args.oos_start,
            oos_end=args.oos_end,
        )
        result = train_advanced_model(dataset, factors, config)
        write_advanced_outputs(result, args.out)
    except ModelUnavailableError as exc:
        print(f"进阶模型不可用：{exc}", file=sys.stderr)
        return 1
    except (DataSourceError, ValueError, OSError) as exc:
        print(f"进阶模型失败：{exc}", file=sys.stderr)
        return 1

    baseline = result.metrics["baseline"]
    hgb = result.metrics["hgb"]
    payload = {
        "ok": True,
        "out": str(args.out),
        "train_rows": result.metrics["train_rows"],
        "oos_rows": result.metrics["oos_rows"],
        "train_label_end": result.metrics["train_label_end"],
        "baseline_ic20": baseline["oos_ic"]["composite"]["20"]["mean_ic"],
        "hgb_ic20": hgb["oos_ic"]["composite"]["20"]["mean_ic"],
        "delta": result.metrics["delta"],
        "baseline_long_short": baseline["long_short"],
        "hgb_long_short": hgb["long_short"],
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"进阶模型完成：{args.out}")
        print(
            f"训练样本 {payload['train_rows']} 条，样本外 {payload['oos_rows']} 条，"
            f"训练标签截止 {payload['train_label_end']}"
        )
        print(f"复合 20 日 OOS IC: {payload['baseline_ic20']:.4f} -> {payload['hgb_ic20']:.4f}")
        base_ls = payload["baseline_long_short"]
        hgb_ls = payload["hgb_long_short"]
        print(f"多空年化: {base_ls['annualized_return']:.2%} -> {hgb_ls['annualized_return']:.2%}")
        print(f"多空 Sharpe: {base_ls['sharpe']:.2f} -> {hgb_ls['sharpe']:.2f}")
        print(f"多空最大回撤: {base_ls['max_drawdown']:.2%} -> {hgb_ls['max_drawdown']:.2%}")
    return 0


def _drift_monitor(args: argparse.Namespace) -> int:
    try:
        dataset = CsvDataAdapter(args.data_dir).load()
        factors = read_fused_factors(args.factors)
        config = DriftConfig(
            horizon=args.horizon,
            window=args.window,
            train_start=args.train_start,
            train_end=args.train_end,
            oos_start=args.oos_start,
            oos_end=args.oos_end,
            score_column=args.score_column,
        )
        result = compute_drift(dataset, factors, config)
        write_drift_outputs(result, args.out)
    except (DataSourceError, ValueError, OSError) as exc:
        print(f"漂移监控失败：{exc}", file=sys.stderr)
        return 1

    payload = {
        "ok": True,
        "out": str(args.out),
        "latest": result.metrics["latest"],
        "train": result.metrics["train"],
        "psi": result.metrics["psi"],
        "severity": result.metrics["severity"],
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        severity = payload["severity"]
        print(f"漂移监控完成：{args.out}")
        print(
            f"最新滚动 IC: {payload['latest']['rolling_ic']:.4f}，"
            f"训练期均值: {payload['train']['ic_mean']:.4f}"
        )
        print(
            f"PSI 最大值: {payload['psi']['max']:.4f}，"
            f"IC 严重度: {severity['ic']:.2f}，"
            f"分布严重度: {severity['distribution']:.2f}"
        )
        print(f"综合漂移严重度: {severity['overall']:.2f} ({severity['level']})")
    return 0


def _pipeline(args: argparse.Namespace) -> int:
    try:
        result = run_pipeline(
            PipelineConfig(
                root=args.root,
                seed=args.seed,
                start=args.start,
                end=args.end,
                train_start=args.train_start,
                train_end=args.train_end,
                oos_start=args.oos_start,
                oos_end=args.oos_end,
                factor_start=args.factor_start,
                sector_limit=args.sector_limit,
            )
        )
    except (DataSourceError, ValueError, OSError) as exc:
        print(f"流水线失败：{exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("流水线完成")
        print(f"  数据: {result['data_dir']}（{result['rows']['prices']} 行）")
        print(f"  因子: {result['factors_dir']}（{result['rows']['factors']} 行）")
        print(
            "  回测: "
            f"复合 20 日 IC {result['backtest']['ic20']:.4f}，"
            f"多空年化 {result['backtest']['long_short']['annualized_return']:.2%}"
        )
        delta_ic = result["news"]["delta"]["ic_20"]
        delta_text = "n/a" if delta_ic is None else f"{delta_ic:+.4f}"
        print(
            "  新闻: "
            f"同向确认 {result['news']['coverage']['confirmed_rate']:.2%}，"
            f"20 日 IC 增量 {delta_text}"
        )
        advanced = result["advanced"]
        if advanced["skipped"]:
            print(f"  进阶模型: 跳过（{advanced['reason']}）")
        else:
            print(
                "  进阶模型: "
                f"线性 IC {advanced['baseline_ic20']:.4f} -> "
                f"HGB IC {advanced['hgb_ic20']:.4f}"
            )
        print(
            "  漂移: "
            f"{result['drift']['severity']['overall']:.2f} "
            f"({result['drift']['severity']['level']})"
        )
        print("  下一步: python -m future.cli web")
    return 0


def _web(args: argparse.Namespace) -> int:
    try:
        from future.web.app import run_app
    except ImportError as exc:
        print(
            f"Web 依赖未安装：{exc}。请执行 uv pip install -e '.[web]' 后重试",
            file=sys.stderr,
        )
        return 1
    try:
        run_app(
            host=args.host,
            port=args.port,
            debug=args.debug,
            data_root=args.data_root,
        )
    except DataSourceError as exc:
        print(f"页面启动失败：{exc}", file=sys.stderr)
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 主入口，返回进程退出码。"""

    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    try:
        settings = Settings.from_env()
    except ValueError as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 2

    if args.command == "generate-data":
        return _generate_data(args, settings)
    if args.command == "validate-data":
        return _validate_data(args)
    if args.command == "compute-factors":
        return _compute_factors(args)
    if args.command == "backtest":
        return _run_backtest(args)
    if args.command == "fuse-news":
        return _fuse_news(args)
    if args.command == "advanced-model":
        return _advanced_model(args)
    if args.command == "drift-monitor":
        return _drift_monitor(args)
    if args.command == "pipeline":
        return _pipeline(args)
    if args.command == "web":
        return _web(args)
    return _info(args, settings)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
