"""Integration Hub: global workspace, event bus, triple network."""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from integration_hub.global_workspace import GlobalWorkspace, Representation, BroadcastEvent
from integration_hub.event_bus import EventBus, Event
from integration_hub.triple_network import TripleNetworkModel, NetworkState

__all__ = [
    "GlobalWorkspace", "Representation", "BroadcastEvent",
    "EventBus", "Event",
    "TripleNetworkModel", "NetworkState",
]
