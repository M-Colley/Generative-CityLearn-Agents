from __future__ import annotations

import datetime as dt
import logging

LOGGER = logging.getLogger("generative_citylearn_agents")


def _timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def log_info(message: str) -> None:
    formatted = f"[{_timestamp()}] [INFO] {message}"
    LOGGER.info(message)
    print(formatted)


def log_warn(message: str) -> None:
    formatted = f"[{_timestamp()}] [WARN] {message}"
    LOGGER.warning(message)
    print(formatted)
