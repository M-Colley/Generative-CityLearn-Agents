from __future__ import annotations

import time
from typing import Optional


class Timer:
    """Simple wall-clock timer modeled after ``util.time.Timer`` in GTA."""

    def __init__(self) -> None:
        self._start: Optional[float] = None

    def start(self) -> None:
        self._start = time.perf_counter()

    def stop(self) -> str:
        if self._start is None:
            return "0.00s"
        elapsed = time.perf_counter() - self._start
        self._start = None
        return f"{elapsed:.2f}s"
