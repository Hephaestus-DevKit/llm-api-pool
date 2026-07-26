"""Regressions for the ways a streamed response can leak a channel permit.

A leaked permit is unrecoverable: `router.acquire` waits on a BoundedSemaphore with no
timeout, so once `max_concurrent` permits are lost (2 by default for web channels) every
later request on that channel hangs forever.
"""
from __future__ import annotations

import asyncio

import main
import pytest
from conftest import FakeBackend


def anthropic_body(**overrides):
    body = {"model": "claude-sonnet-5", "max_tokens": 64, "stream": True,
            "messages": [{"role": "user", "content": "hi"}]}
    body.update(overrides)
    return body


def permits(channel):
    return main.router.get_sem(channel)._value


def test_a_completed_stream_returns_its_permit(client, make_channel):
    channel, _backend = make_channel("official_claude", max_concurrent=2,
                                     backend=FakeBackend(stream_chunks=[
                                         {"choices": [{"delta": {"content": "hi"}}]}]))
    client.post("/v1/messages", json=anthropic_body())
    assert permits(channel) == 2
    assert main.router.stats[channel["id"]]["in_flight"] == 0


def test_a_failing_stream_returns_its_permit(client, make_channel):
    class Exploding(FakeBackend):
        async def stream(self, req, ch):
            yield {"choices": [{"delta": {"content": "partial"}}]}
            raise RuntimeError("reset")

    channel, _backend = make_channel("official_claude", max_concurrent=2, backend=Exploding())
    client.post("/v1/messages", json=anthropic_body())
    assert permits(channel) == 2


def test_a_stream_cancelled_before_its_first_chunk_returns_its_permit(pool, make_channel):
    """`except Exception` does not catch CancelledError; the permit walked out of the
    function with nothing left to hand it back."""
    class Hanging(FakeBackend):
        async def stream(self, req, ch):
            await asyncio.sleep(60)
            yield {"choices": [{"delta": {"content": "never"}}]}

    channel, _backend = make_channel("official_claude", max_concurrent=2, backend=Hanging())

    async def scenario():
        canon = main.anthropic_to_canonical(anthropic_body())
        task = asyncio.create_task(main.open_stream_with_failover(canon, "/v1/messages"))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert permits(channel) == 2
    assert main.router.stats[channel["id"]]["in_flight"] == 0


def test_a_response_whose_body_is_never_consumed_returns_its_permit(pool, make_channel):
    """Starlette can lose the race between listen_for_disconnect and stream_response, so the
    body generator is never started and its `finally` never runs. The background task is the
    only thing that gives the permit back."""
    channel, _backend = make_channel("official_claude", max_concurrent=2,
                                     backend=FakeBackend(stream_chunks=[
                                         {"choices": [{"delta": {"content": "hi"}}]}]))

    async def scenario():
        canon = main.anthropic_to_canonical(anthropic_body())
        lease = await main.open_stream_with_failover(canon, "/v1/messages")
        response = main.stream_with_lease(lease, main.anthropic_stream_response, "/v1/messages")
        assert permits(channel) == 1, "the lease should be holding a permit"
        await response.background()  # what Starlette runs once the response is done with

    asyncio.run(scenario())
    assert permits(channel) == 2
    assert main.router.stats[channel["id"]]["in_flight"] == 0


def test_finishing_twice_does_not_over_release(pool, make_channel):
    channel, _backend = make_channel("official_claude", max_concurrent=2,
                                     backend=FakeBackend(stream_chunks=[]))

    async def scenario():
        canon = main.anthropic_to_canonical(anthropic_body())
        lease = await main.open_stream_with_failover(canon, "/v1/messages")
        await lease.finish(True, None, 0)
        await lease.abandon()
        await lease.finish(True, None, 0)

    asyncio.run(scenario())
    assert permits(channel) == 2


def test_a_channel_survives_many_streams(client, make_channel):
    """The failure mode is cumulative: leak max_concurrent permits and the channel is dead."""
    channel, _backend = make_channel("official_claude", max_concurrent=2,
                                     backend=FakeBackend(stream_chunks=[
                                         {"choices": [{"delta": {"content": "hi"}}]}]))
    for _ in range(10):
        assert client.post("/v1/messages", json=anthropic_body()).status_code == 200
    assert permits(channel) == 2
