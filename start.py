#!/usr/bin/env python
"""项目根目录启动入口：python start.py。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from future.start import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(project_root_path=ROOT))
