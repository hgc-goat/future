"""一键启动脚本的核心逻辑。

行为：
1. 若项目根目录存在 `.venv`，自动切换到其中的 Python；
2. 若页面 artifacts 缺失，自动运行完整 pipeline；
3. 启动 Dash 服务并可选自动打开浏览器。
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

REQUIRED_ARTIFACTS = (
    "data/mock/prices.csv",
    "data/factors_news/fused_factors.csv",
    "artifacts/backtest/metrics.json",
    "artifacts/news/analysis.json",
    "artifacts/advanced/metrics.json",
    "artifacts/drift/drift.json",
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def artifacts_ready(root: Path) -> bool:
    return all((root / relative).is_file() for relative in REQUIRED_ARTIFACTS)


def _venv_python(root: Path) -> Path | None:
    for relative in (".venv/Scripts/python.exe", ".venv/bin/python"):
        candidate = root / relative
        if candidate.is_file():
            return candidate
    return None


def _run_with_venv(root: Path) -> int | None:
    if os.getenv("FUTURE_NO_REEXEC") == "1":
        return None
    venv = _venv_python(root)
    if venv is None:
        return None
    current = Path(sys.executable).resolve()
    target = venv.resolve()
    if current == target:
        return None
    script = Path(sys.argv[0]).resolve()
    print(f"切换到项目虚拟环境：{venv}")
    return subprocess.run(
        [str(target), str(script), *sys.argv[1:]],
        cwd=str(script.parent),
        check=False,
    ).returncode


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _build_artifacts(root: Path, seed: int) -> bool:
    missing_modules = [module for module in ("pandas", "sklearn") if not _has_module(module)]
    if missing_modules:
        print(
            "缺少依赖："
            + ", ".join(missing_modules)
            + '\n请先安装：uv pip install -e ".[dev,ml,web]"',
            file=sys.stderr,
        )
        return False

    from future.pipeline import PipelineConfig, run_pipeline

    print("页面产物缺失，开始生成数据、因子、回测、新闻、HGB 与漂移结果…")
    try:
        result = run_pipeline(PipelineConfig(root=root, seed=seed))
    except Exception as exc:  # noqa: BLE001 - 启动脚本需要统一给出可读错误
        print(f"产物生成失败：{exc}", file=sys.stderr)
        return False

    print(
        "产物生成完成："
        f"{result['rows']['prices']} 行行情，"
        f"{result['rows']['factors']} 行因子，"
        f"漂移严重度 {result['drift']['severity']['overall']:.2f}"
    )
    return True


def _ensure_web_dependencies() -> bool:
    missing_modules = [module for module in ("pandas", "dash", "plotly") if not _has_module(module)]
    if missing_modules:
        print(
            "缺少 Web 依赖："
            + ", ".join(missing_modules)
            + '\n请先安装：uv pip install -e ".[dev,ml,web]"',
            file=sys.stderr,
        )
        return False
    return True


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="启动 future 板块趋势概率系统",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--rebuild", action="store_true", help="强制重新生成全部产物")
    parser.add_argument("--build-only", action="store_true", help="只生成产物，不启动服务")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", type=Path, default=None)
    return parser


def main(
    argv: list[str] | None = None,
    project_root_path: Path | None = None,
) -> int:
    root = Path(project_root_path or project_root())
    reexec_code = _run_with_venv(root)
    if reexec_code is not None:
        return reexec_code

    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.data_root is not None:
        root = args.data_root

    needs_build = args.rebuild or not artifacts_ready(root)
    if needs_build and not _build_artifacts(root, args.seed):
        return 1
    if args.build_only:
        print("仅生成产物，未启动 Web 服务。")
        return 0
    if not _ensure_web_dependencies():
        return 1

    from future.data.base import DataSourceError
    from future.web.app import run_app

    url = f"http://{args.host}:{args.port}"
    print(f"启动 future：{url}")
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    try:
        run_app(
            host=args.host,
            port=args.port,
            debug=args.debug,
            data_root=root,
        )
    except OSError as exc:
        print(f"端口启动失败：{exc}\n可改用 --port 8051 重试。", file=sys.stderr)
        return 1
    except DataSourceError as exc:
        print(f"页面数据加载失败：{exc}", file=sys.stderr)
        return 1
    return 0


__all__ = ["artifacts_ready", "main", "project_root"]
