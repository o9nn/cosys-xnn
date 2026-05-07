"""LIF neuron model, neural populations, and synaptic connectivity."""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import List


class NeuronType(Enum):
    EXCITATORY = "excitatory"
    INHIBITORY = "inhibitory"
    MODULATORY = "modulatory"


@dataclass
class LeakyIntegrateFireNeuron:
    """τ dV/dt = -(V - V_rest) + R·I"""
    tau: float = 20.0
    v_rest: float = -70.0
    v_reset: float = -75.0
    v_threshold: float = -50.0
    resistance: float = 10.0
    v: float = field(default=-70.0)
    spike: bool = field(default=False)

    def step(self, input_current: float, dt: float = 1.0) -> bool:
        dv = (-(self.v - self.v_rest) + self.resistance * input_current) / self.tau
        self.v += dv * dt
        if self.v >= self.v_threshold:
            self.v = self.v_reset
            self.spike = True
            return True
        self.spike = False
        return False


@dataclass
class NeuralPopulation:
    name: str
    n_neurons: int
    neuron_type: NeuronType
    neurons: List[LeakyIntegrateFireNeuron] = field(default_factory=list)

    def __post_init__(self):
        self.neurons = [LeakyIntegrateFireNeuron() for _ in range(self.n_neurons)]
        self.activity = np.zeros(self.n_neurons)

    def step(self, input_currents: np.ndarray, dt: float = 1.0) -> np.ndarray:
        spikes = np.zeros(self.n_neurons)
        for i, neuron in enumerate(self.neurons):
            spikes[i] = float(neuron.step(input_currents[i], dt))
        self.activity = 0.9 * self.activity + 0.1 * spikes
        return spikes

    def get_firing_rate(self) -> float:
        return float(np.mean(self.activity))


class SynapticMatrix:
    """Weight matrix connecting two neural populations."""

    def __init__(self, n_pre: int, n_post: int, weight_scale: float = 0.1):
        self.weights = np.random.randn(n_pre, n_post) * weight_scale
        self.n_pre = n_pre
        self.n_post = n_post

    def forward(self, pre_spikes: np.ndarray) -> np.ndarray:
        """Compute post-synaptic currents from pre-synaptic spikes."""
        return pre_spikes @ self.weights

    def apply_hebbian(
        self,
        pre_spikes: np.ndarray,
        post_spikes: np.ndarray,
        learning_rate: float = 0.01,
    ) -> None:
        """Hebbian weight update."""
        delta = learning_rate * np.outer(pre_spikes, post_spikes)
        self.weights += delta
        self.weights = np.clip(self.weights, -2.0, 2.0)
