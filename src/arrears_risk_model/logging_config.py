"""Logging setup for the arrears risk model.

Single configure-once function that other entry points (train.py,
predict.py) call before any work begins. Format is line-oriented and
easy to grep — sufficient for batch jobs producing periodic priority
lists, where the alternative (JSON for log aggregators) would be
overkill given there is no live serving stack.
"""

from __future__ import annotations

import logging
import sys

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


def configure_logging(level: int | str = logging.INFO) -> None:
    """Configure the root logger for stderr output. Idempotent."""
    root = logging.getLogger()
    # Clear existing handlers so re-invocation doesn't duplicate output.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT))
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Convenience wrapper. Use `get_logger(__name__)` in modules."""
    return logging.getLogger(name)
