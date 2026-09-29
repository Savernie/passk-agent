import asyncio
from dataclasses import dataclass
from enum import Enum


class IdempotencyStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


@dataclass
class IdempotencyRecord:
    status: IdempotencyStatus
    result: object = None


class IdempotencyConflict(Exception):
    """Raised when a key is already in progress — the caller must not blindly retry."""


class IdempotencyStore:
    def __init__(self):
        self._records: dict[str, IdempotencyRecord] = {}
        self._lock = asyncio.Lock()

    async def begin(self, key: str) -> IdempotencyRecord | None:
        async with self._lock:
            existing = self._records.get(key)
            if existing is not None:
                return existing
            self._records[key] = IdempotencyRecord(status=IdempotencyStatus.IN_PROGRESS)
            return None

    async def complete(self, key: str, result):
        async with self._lock:
            self._records[key] = IdempotencyRecord(
                status=IdempotencyStatus.COMPLETED, result=result
            )


async def run_idempotent(store: IdempotencyStore, key: str, fn):
    existing = await store.begin(key)
    if existing is not None:
        if existing.status == IdempotencyStatus.COMPLETED:
            return existing.result
        raise IdempotencyConflict(f"key {key!r} already in progress")

    result = await fn()
    await store.complete(key, result)
    return result