"""支持 `python -m future` 调用 CLI。"""

from future.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
