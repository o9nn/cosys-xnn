"""Global Workspace Theory implementation (Dehaene & Changeux)."""

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time


@dataclass
class Representation:
    source: str
    content: Any
    salience: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class BroadcastEvent:
    representation: Representation
    recipients: List[str]
    timestamp: float = field(default_factory=time.time)


class GlobalWorkspace:
    """Winner-take-all competition for conscious access."""

    def __init__(self, ignition_threshold: float = 0.7):
        self.ignition_threshold = ignition_threshold
        self.connected_services: List[str] = []
        self.broadcast_history: List[BroadcastEvent] = []
        self._attention_weights: Dict[str, float] = {}

    def register_service(self, service_name: str, attention_weight: float = 1.0) -> None:
        self.connected_services.append(service_name)
        self._attention_weights[service_name] = attention_weight

    def set_attention_weight(self, service_name: str, weight: float) -> None:
        self._attention_weights[service_name] = max(0.0, min(1.0, weight))

    def compete_for_access(
        self, representations: List[Representation]
    ) -> Optional[BroadcastEvent]:
        """Bottom-up salience × top-down attention → winner-take-all selection."""
        if not representations:
            return None
        activations = []
        for rep in representations:
            attention = self._attention_weights.get(rep.source, 1.0)
            activations.append(rep.salience * attention)
        winner_idx = int(np.argmax(activations))
        if activations[winner_idx] >= self.ignition_threshold:
            return self.broadcast(representations[winner_idx])
        return None

    def broadcast(self, representation: Representation) -> BroadcastEvent:
        event = BroadcastEvent(
            representation=representation,
            recipients=list(self.connected_services),
        )
        self.broadcast_history.append(event)
        if len(self.broadcast_history) > 1000:
            self.broadcast_history.pop(0)
        return event

    def get_last_broadcast(self) -> Optional[BroadcastEvent]:
        return self.broadcast_history[-1] if self.broadcast_history else None
