"""M-1: Hypothalamus (PVN, LH, VMH) — homeostatic monitoring, autonomic control."""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from typing import Optional

from cosmos_core import BaseCosmosService, ServiceConfig, ServiceMessage, create_message
from neural_models.lif_neuron import NeuralPopulation, NeuronType


class HypothalamusService(BaseCosmosService):
    """M-1: Hypothalamus nuclei. Multiple peptides."""

    def __init__(self, config: ServiceConfig, n_neurons: int = 60):
        super().__init__(config)
        self.population = NeuralPopulation("Hypothalamus", n_neurons, NeuronType.MODULATORY)
        self.homeostatic_setpoints = {"arousal": 0.5, "stress": 0.3, "energy": 0.7}

    async def initialize(self) -> None:
        self.log("info", "Hypothalamus Service initialized")
        self.initialized = True

    async def process(self, message: ServiceMessage) -> Optional[ServiceMessage]:
        input_currents = np.random.randn(self.population.n_neurons) * 2.0
        spikes = self.population.step(input_currents)
        firing_rate = self.population.get_firing_rate()
        homeostatic_state = {
            "arousal_level": firing_rate,
            "stress_level": max(0, firing_rate - 0.5),
            "energy_level": 1.0 - firing_rate,
            "autonomic_balance": "sympathetic" if firing_rate > 0.5 else "parasympathetic",
        }
        return create_message(
            "HOMEOSTATIC_STATE", homeostatic_state, self.config.service_name
        )

    async def shutdown(self) -> None:
        self.log("info", "Hypothalamus Service shutdown")
