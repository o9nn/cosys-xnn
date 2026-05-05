"""Triple Network Model: DMN, Salience Network, Central Executive Network."""

from dataclasses import dataclass, field
from typing import Any, Dict, List
import numpy as np


@dataclass
class NetworkState:
    name: str
    active: bool = False
    activation_level: float = 0.0
    history: List[float] = field(default_factory=list)

    def activate(self, level: float = 1.0) -> None:
        self.active = True
        self.activation_level = level
        self._record(level)

    def suppress(self) -> None:
        self.active = False
        self.activation_level *= 0.1
        self._record(self.activation_level)

    def monitor(self) -> None:
        self._record(self.activation_level)

    def _record(self, v: float) -> None:
        self.history.append(v)
        if len(self.history) > 200:
            self.history.pop(0)


class TripleNetworkModel:
    """
    Triple network model of brain function (Menon, 2011).
    Salience Network mediates switching between DMN and CEN.
    """

    def __init__(self, switch_threshold: float = 0.5):
        self.dmn = NetworkState("DefaultModeNetwork")
        self.sn = NetworkState("SalienceNetwork")
        self.cen = NetworkState("CentralExecutiveNetwork")
        self.switch_threshold = switch_threshold
        self.dmn.activate(0.5)  # DMN active at rest

    def switch_networks(self, salience_signal: float) -> Dict[str, Any]:
        """SN mediates switching: high salience → CEN; low salience → DMN."""
        self.sn.monitor()
        self.sn.activation_level = salience_signal

        if salience_signal > self.switch_threshold:
            cen_level = min(1.0, salience_signal)
            self.cen.activate(cen_level)
            self.dmn.suppress()
            mode = "task_focused"
        else:
            dmn_level = min(1.0, 1.0 - salience_signal)
            self.dmn.activate(dmn_level)
            self.cen.suppress()
            mode = "mind_wandering"

        return {
            "mode": mode,
            "salience": salience_signal,
            "dmn": self.dmn.activation_level,
            "sn": self.sn.activation_level,
            "cen": self.cen.activation_level,
        }

    def get_state(self) -> Dict[str, Any]:
        return {
            "dmn_active": self.dmn.active,
            "sn_active": self.sn.active,
            "cen_active": self.cen.active,
            "dmn_level": self.dmn.activation_level,
            "sn_level": self.sn.activation_level,
            "cen_level": self.cen.activation_level,
        }
