#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Наблюдатель drop-папки. Предпочтительно: pip install -e . && meet2task-watch"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from meet2task.watch_drop import main

if __name__ == "__main__":
    main()
