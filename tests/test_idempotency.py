import pytest
from idempotency import IdempotencyStore, IdempotencyConflict, run_idempotent


@pytest.mark.asyncio
async def test_retry_does_not_double_spend_after_crash_before_commit():
    store = IdempotencyStore()
    spend_count = {"count": 0}

    async def flaky_expensive_call():
        spend_count["count"] += 1  # the money is spent here
        raise RuntimeError("simulated crash before DB commit")  # then it crashes

    key = "run-1:task-3:trial-0"

    with pytest.raises(RuntimeError):
        await run_idempotent(store, key, flaky_expensive_call)
    assert spend_count["count"] == 1

    # Retry with the SAME key after the crash:
    with pytest.raises(IdempotencyConflict):
        await run_idempotent(store, key, flaky_expensive_call)

    assert spend_count["count"] == 1, "retry must not re-spend for an in-progress key"