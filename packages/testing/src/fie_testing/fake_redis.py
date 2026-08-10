"""A functional in-memory Redis double.

This is not a mock that returns canned values — it implements the key/value,
expiry, and stream consumer-group semantics the platform actually relies on,
including pending-entry lists and delivery counts. That is what lets the event
bus's dead-letter and reclaim paths be unit-tested deterministically.

Integration tests still run the same logic against real Redis; this double
exists so those paths are covered even when infrastructure is unavailable, not
to replace them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from redis.exceptions import ResponseError


@dataclass
class _PendingEntry:
    consumer: str
    delivered_count: int
    delivered_at_ms: float


@dataclass
class _GroupState:
    last_delivered_index: int = 0
    pending: dict[str, _PendingEntry] = field(default_factory=dict)


class FakeRedis:
    """In-memory stand-in for ``redis.asyncio.Redis``.

    Time is virtual: call :meth:`advance` to age pending entries and expire
    keys instead of sleeping.
    """

    def __init__(self) -> None:
        self._kv: dict[str, str] = {}
        self._expiry_ms: dict[str, float] = {}
        self._streams: dict[str, list[tuple[str, dict[str, str]]]] = {}
        self._groups: dict[tuple[str, str], _GroupState] = {}
        self._now_ms: float = 0.0
        self._sequence = 0
        self.closed = False

    # -- virtual clock -------------------------------------------------------

    def advance(self, milliseconds: float) -> None:
        """Move the virtual clock forward."""
        self._now_ms += milliseconds

    def _purge_expired(self) -> None:
        expired = [key for key, at in self._expiry_ms.items() if at <= self._now_ms]
        for key in expired:
            self._kv.pop(key, None)
            self._expiry_ms.pop(key, None)

    # -- key/value -----------------------------------------------------------

    async def get(self, key: str) -> str | None:
        self._purge_expired()
        return self._kv.get(key)

    async def set(
        self,
        key: str,
        value: str,
        *,
        ex: int | None = None,
        nx: bool = False,
    ) -> bool | None:
        self._purge_expired()
        if nx and key in self._kv:
            return None
        self._kv[key] = value
        if ex is not None:
            self._expiry_ms[key] = self._now_ms + ex * 1000
        else:
            self._expiry_ms.pop(key, None)
        return True

    async def delete(self, *keys: str) -> int:
        self._purge_expired()
        removed = 0
        for key in keys:
            if self._kv.pop(key, None) is not None:
                self._expiry_ms.pop(key, None)
                removed += 1
        return removed

    async def exists(self, key: str) -> int:
        self._purge_expired()
        return 1 if key in self._kv else 0

    async def incrby(self, key: str, amount: int = 1) -> int:
        self._purge_expired()
        current = int(self._kv.get(key, "0"))
        current += amount
        self._kv[key] = str(current)
        return current

    async def expire(self, key: str, seconds: int) -> bool:
        if key not in self._kv:
            return False
        self._expiry_ms[key] = self._now_ms + seconds * 1000
        return True

    async def ttl(self, key: str) -> int:
        if key not in self._kv:
            return -2
        if key not in self._expiry_ms:
            return -1
        return int((self._expiry_ms[key] - self._now_ms) / 1000)

    async def eval(self, script: str, numkeys: int, *args: Any) -> int:
        """Support the ownership-checked lock release script."""
        keys = list(args[:numkeys])
        argv = list(args[numkeys:])
        if "del" in script and keys and argv:
            if self._kv.get(keys[0]) == argv[0]:
                await self.delete(keys[0])
                return 1
            return 0
        raise NotImplementedError("FakeRedis.eval supports only the lock-release script")

    async def ping(self) -> bool:
        return True

    async def aclose(self) -> None:
        self.closed = True

    # -- streams -------------------------------------------------------------

    def _next_id(self) -> str:
        self._sequence += 1
        return f"{self._sequence}-0"

    async def xadd(
        self,
        stream: str,
        fields: dict[str, Any],
        *,
        maxlen: int | None = None,
        approximate: bool = True,
    ) -> str:
        entries = self._streams.setdefault(stream, [])
        message_id = self._next_id()
        entries.append((message_id, {str(k): str(v) for k, v in fields.items()}))
        if maxlen is not None and len(entries) > maxlen:
            del entries[: len(entries) - maxlen]
        return message_id

    async def xgroup_create(
        self, stream: str, group: str, *, id: str = "0", mkstream: bool = False
    ) -> bool:
        if stream not in self._streams:
            if not mkstream:
                raise ResponseError("NOGROUP No such key")
            self._streams[stream] = []
        if (stream, group) in self._groups:
            raise ResponseError("BUSYGROUP Consumer Group name already exists")
        start = 0 if id == "0" else len(self._streams[stream])
        self._groups[(stream, group)] = _GroupState(last_delivered_index=start)
        return True

    async def xreadgroup(
        self,
        group: str,
        consumer: str,
        streams: dict[str, str],
        *,
        count: int = 10,
        block: int | None = None,
    ) -> list[tuple[str, list[tuple[str, dict[str, str]]]]]:
        response: list[tuple[str, list[tuple[str, dict[str, str]]]]] = []
        for stream in streams:
            state = self._groups.get((stream, group))
            if state is None:
                raise ResponseError("NOGROUP No such consumer group")
            entries = self._streams.get(stream, [])
            batch = entries[state.last_delivered_index : state.last_delivered_index + count]
            if not batch:
                continue
            state.last_delivered_index += len(batch)
            for message_id, _fields in batch:
                state.pending[message_id] = _PendingEntry(
                    consumer=consumer, delivered_count=1, delivered_at_ms=self._now_ms
                )
            response.append((stream, batch))
        return response

    async def xack(self, stream: str, group: str, *message_ids: str) -> int:
        state = self._groups.get((stream, group))
        if state is None:
            return 0
        return sum(1 for mid in message_ids if state.pending.pop(mid, None) is not None)

    async def xpending_range(
        self,
        stream: str,
        group: str,
        *,
        min: str = "-",
        max: str = "+",
        count: int = 10,
        idle: int | None = None,
    ) -> list[dict[str, Any]]:
        state = self._groups.get((stream, group))
        if state is None:
            return []
        results: list[dict[str, Any]] = []
        for message_id, entry in list(state.pending.items())[:count]:
            age = self._now_ms - entry.delivered_at_ms
            if idle is not None and age < idle:
                continue
            results.append(
                {
                    "message_id": message_id,
                    "consumer": entry.consumer,
                    "time_since_delivered": int(age),
                    "times_delivered": entry.delivered_count,
                }
            )
        return results

    async def xclaim(
        self,
        stream: str,
        group: str,
        consumer: str,
        *,
        min_idle_time: int,
        message_ids: list[str],
    ) -> list[tuple[str, dict[str, str]]]:
        state = self._groups.get((stream, group))
        if state is None:
            return []
        entries = dict(self._streams.get(stream, []))
        claimed: list[tuple[str, dict[str, str]]] = []
        for message_id in message_ids:
            pending = state.pending.get(message_id)
            if pending is None:
                continue
            if (self._now_ms - pending.delivered_at_ms) < min_idle_time:
                continue
            pending.consumer = consumer
            pending.delivered_count += 1
            pending.delivered_at_ms = self._now_ms
            if message_id in entries:
                claimed.append((message_id, entries[message_id]))
        return claimed

    async def xrange(
        self, stream: str, *, min: str = "-", max: str = "+"
    ) -> list[tuple[str, dict[str, str]]]:
        entries = self._streams.get(stream, [])
        if min == "-" and max == "+":
            return list(entries)
        return [(mid, fields) for mid, fields in entries if min <= mid <= max]

    async def xlen(self, stream: str) -> int:
        return len(self._streams.get(stream, []))

    # -- test introspection --------------------------------------------------

    def stream_entries(self, stream: str) -> list[tuple[str, dict[str, str]]]:
        return list(self._streams.get(stream, []))

    def pending_count(self, stream: str, group: str) -> int:
        state = self._groups.get((stream, group))
        return len(state.pending) if state else 0

    def set_delivery_count(self, stream: str, group: str, message_id: str, count: int) -> None:
        """Force a delivery count, to drive the dead-letter threshold."""
        state = self._groups[(stream, group)]
        state.pending[message_id].delivered_count = count


__all__ = ["FakeRedis"]
