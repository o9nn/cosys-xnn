"""O-4: Substantia Nigra (SNc/SNr) — dopaminergic reward modulation."""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from typing import Optional

from cosmos_core import BaseCosmosService, ServiceConfig, ServiceMessage, create_message
from neural_models.lif_neuron import NeuralPopulation, NeuronType


class SubstantiaNigraService(BaseCosmosService):
    """O-4: Substantia Nigra. Dopaminergic output for reward prediction and motor control."""

    def __init__(self, config: ServiceConfig, n_neurons: int = 60):
        super().__init__(config)
        self.population = NeuralPopulation("Substantia Nigra", n_neurons, NeuronType.MODULATORY)
        self.dopamine_level = 0.5
        self.reward_history: list = []

    async def initialize(self) -> None:
        self.log("info", "Substantia Nigra Service initialized")
        self.initialized = True

    async def process(self, message: ServiceMessage) -> Optional[ServiceMessage]:
        if message.type in ("MOTOR_GATING", "ACTION_SELECTED", "REWARD_SIGNAL"):
            input_currents = np.random.randn(self.population.n_neurons) * 4.0
            spikes = self.population.step(input_currents)
            firing_rate = self.population.get_firing_rate()
            rpe = firing_rate - self.dopamine_level
            self.dopamine_level = 0.9 * self.dopamine_level + 0.1 * firing_rate
            self.reward_history.append(rpe)
            if len(self.reward_history) > 50:
                self.reward_history.pop(0)
            result = {
                "dopamine_level": self.dopamine_level,
                "reward_prediction_error": rpe,
                "snc_activity": firing_rate,
                "snr_activity": max(0.0, 1.0 - firing_rate),
                "reward_trend": (
                    float(np.mean(self.reward_history[-10:]))
                    if self.reward_history
                    else 0.0
                ),
            }
            return create_message("DOPAMINE_SIGNAL", result, self.config.service_name)
        return None

    async def shutdown(self) -> None:
        self.log("info", "Substantia Nigra Service shutdown")
