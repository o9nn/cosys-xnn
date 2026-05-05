"""Async event bus with neural pathway constraints and timing simulation."""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List

# Inter-region timing constraints (ms)
TIMING_MS = {
    "synaptic":       (1,   5),
    "local":          (10,  50),
    "inter_regional": (50,  200),
    "global":         (200, 500),
}

PATHWAY_TIMING = {
    "cortico_striatal":     "inter_regional",
    "basal_ganglia_limbic": "inter_regional",
    "limbic_prefrontal":    "global",
    "intra_triad":          "local",
    "synaptic":             "synaptic",
}


@dataclass
class Event:
    topic: str
    payload: Any
    source: str
    pathway: str = "intra_triad"
    timestamp: float = field(default_factory=time.time)


HandlerType = Callable[[Event], Awaitable[None]]


class EventBus:
    """Async publish/subscribe bus with biologically-realistic timing delays."""

    def __init__(self, simulate_timing: bool = False):
        self._subscribers: Dict[str, List[HandlerType]] = {}
        self.simulate_timing = simulate_timing
        self._event_log: List[Event] = []

    def subscribe(self, topic: str, handler: HandlerType) -> None:
        self._subscribers.setdefault(topic, []).append(handler)

    async def publish(self, event: Event) -> None:
        self._event_log.append(event)
        if len(self._event_log) > 10000:
            self._event_log.pop(0)

        if self.simulate_timing:
            import random

            timing_key = PATHWAY_TIMING.get(event.pathway, "local")
            lo, hi = TIMING_MS[timing_key]
            delay_ms = random.uniform(lo, hi)
            await asyncio.sleep(delay_ms / 1000.0)

        handlers = self._subscribers.get(event.topic, [])
        await asyncio.gather(*(h(event) for h in handlers), return_exceptions=True)

    def get_log(self, n: int = 100) -> List[Event]:
        return self._event_log[-n:]

    def clear_log(self) -> None:
        self._event_log.clear()
