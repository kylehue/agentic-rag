from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

_SENTINEL = object()


@dataclass(frozen=True)
class Event:
    """A neutral event: a wire name plus a payload dict.

    Domain layers define their own event names; this type is shared.
    """

    name: str
    payload: dict[str, Any] = field(default_factory=dict)


class EventBus:
    """A replayable, per-subscription event bus.

    `publish` is synchronous (no await), so in the single-threaded event loop
    a subscriber registers atomically with respect to publishing: it replays
    exactly the events buffered before it subscribed, then tails the live
    ones, with no gaps or duplicates. When an event whose name is in `terminal`
    is published, the stream ends (subscribers receive it, then stop). With an
    empty `terminal` the stream never ends on its own.
    """

    def __init__(self, terminal: frozenset[str] = frozenset()) -> None:
        self._terminal = terminal
        self._events: list[Event] = []
        self._subscribers: list[asyncio.Queue] = []
        self._finished = False

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def events(self) -> list[Event]:
        """The buffered events (a copy), for snapshot/replay readers."""
        return list(self._events)

    def publish(self, event: Event) -> None:
        self._events.append(event)
        for queue in self._subscribers:
            queue.put_nowait(event)
        if event.name in self._terminal:
            self._finished = True
            for queue in self._subscribers:
                queue.put_nowait(_SENTINEL)

    def _register(self) -> tuple[asyncio.Queue, int]:
        # Synchronous on purpose: atomic with respect to `publish`.
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.append(queue)
        return queue, len(self._events)

    async def subscribe(self) -> AsyncIterator[Event]:
        queue, start = self._register()
        try:
            for event in self._events[:start]:
                yield event
            if self._finished:
                return
            while True:
                item = await queue.get()
                if item is _SENTINEL:
                    break
                yield item
        finally:
            self._subscribers.remove(queue)
