"""
NeuralSystem5StateMachine — 60-step deterministic cycle.
LCM(3, 20) = 60 steps for full Universal-Particular synchronization.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List

# Neural correlate timing constants (ms)
TIMING = {
    "U1_to_U2": 100,
    "U2_to_U3": 200,
    "U3_to_U1": 150,
    "P1_to_P2": 50,
    "P2_to_P3": 100,
    "P3_to_P4": 150,
}

CYCLE_LENGTH = 60  # LCM(3, 20)


@dataclass
class UniversalState:
    name: str
    state: int = 0  # 0–2 (3 universal phases)


@dataclass
class ParticularState:
    name: str
    state: int = 0  # 0–3 (mod 4)


class NeuralSystem5StateMachine:
    """60-step deterministic cycle for neural processing."""

    def __init__(self):
        self.U1 = UniversalState("GlobalWorkspace")
        self.U2 = UniversalState("SalienceNetwork")
        self.U3 = UniversalState("DefaultMode")

        self.P1 = ParticularState("SensoryProcessing")
        self.P2 = ParticularState("AssociationProcessing")
        self.P3 = ParticularState("PrefrontalProcessing")
        self.P4 = ParticularState("MotorProcessing")

        self.t = 0
        self.history: List[Dict[str, Any]] = []

    @property
    def _particulars(self):
        return [self.P1, self.P2, self.P3, self.P4]

    @property
    def _universals(self):
        return [self.U1, self.U2, self.U3]

    def step(self) -> Dict[str, Any]:
        """Execute one step of the 60-step cycle. Returns state snapshot."""
        u_idx = self.t % 3
        self._transition_universal(u_idx)

        p_idx = self.t % 5
        if p_idx < 4:
            self._transition_particular(p_idx, u_idx)

        snapshot = self._snapshot()
        self.history.append(snapshot)
        if len(self.history) > CYCLE_LENGTH:
            self.history.pop(0)
        self.t += 1
        return snapshot

    def run_cycle(self) -> List[Dict[str, Any]]:
        """Run a full 60-step cycle and return all snapshots."""
        return [self.step() for _ in range(CYCLE_LENGTH)]

    def _transition_universal(self, u_idx: int) -> None:
        u = self._universals[u_idx]
        u.state = (u.state + 1) % 3

    def _transition_particular(self, p_idx: int, u_idx: int) -> None:
        """S_i(t+1) = (S_i(t) + Σ_{j≠i} S_j(t) + U_idx(t)) mod 4"""
        active = self._particulars[p_idx]
        others = [s for i, s in enumerate(self._particulars) if i != p_idx]
        conv_sum = sum(s.state for s in others)
        u_state = self._universals[u_idx].state
        active.state = (active.state + conv_sum + u_state) % 4

    def _snapshot(self) -> Dict[str, Any]:
        return {
            "t": self.t,
            "U1": self.U1.state, "U2": self.U2.state, "U3": self.U3.state,
            "P1": self.P1.state, "P2": self.P2.state,
            "P3": self.P3.state, "P4": self.P4.state,
            "u_phase": self.t % 3,
            "p_phase": self.t % 5,
        }

    def get_active_universal(self) -> UniversalState:
        return self._universals[self.t % 3]

    def get_active_particular(self):
        p_idx = self.t % 5
        return self._particulars[p_idx] if p_idx < 4 else None
