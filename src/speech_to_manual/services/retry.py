from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TypeVar

from speech_to_manual.domain.errors import LlmGenerationError

T = TypeVar("T")


class RetryExecutor:
    def __init__(self, retries: int, retry_delay: int, logger: logging.Logger) -> None:
        self._retries = retries
        self._retry_delay = retry_delay
        self._logger = logger

    def run(self, stage_name: str, fn: Callable[[], T]) -> T:
        last_exc: Exception | None = None
        for attempt in range(1, self._retries + 1):
            try:
                return fn()
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                last_exc = exc
                self._logger.warning("%s | attempt %s/%s failed: %s", stage_name, attempt, self._retries, exc)
                if attempt < self._retries:
                    time.sleep(self._retry_delay)
        raise LlmGenerationError(f"{stage_name} failed after {self._retries} attempts: {last_exc}")
