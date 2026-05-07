"""O-4: Cingulate (Autonomic) — visceromotor output, autonomic organization."""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from typing import Optional

from cosmos_core import BaseCosmosService, ServiceConfig, ServiceMessage, create_message
from neural_models.lif_neuron import NeuralPopulation, NeuronType


class CingulateAutonomicService(BaseCosmosService):
    """O-4: Anterior Cingulate Cortex autonomic division. Coordinates visceromotor output."""

    def __init__(self, config: ServiceConfig, n_neurons: int = 70):
        super().__init__(config)
        self.population = NeuralPopulation(
            "Autonomic Cingulate", n_neurons, NeuronType.EXCITATORY
        )

    async def initialize(self) -> None:
        self.log("info", "Cingulate Autonomic Service initialized")
        self.initialized = True

    async def process(self, message: ServiceMessage) -> Optional[ServiceMessage]:
        if message.type in (
            "INTEROCEPTIVE_STATE",
            "EMOTIONAL_STATE",
            "AROUSAL_SIGNAL",
        ):
            input_currents = np.random.randn(self.population.n_neurons) * 3.0
            spikes = self.population.step(input_currents)
            firing_rate = self.population.get_firing_rate()
            result = {
                "visceromotor_output": firing_rate * 10,
                "autonomic_coordination": firing_rate,
                "heart_rate_modulation": firing_rate * 5,
                "respiratory_modulation": (1.0 - firing_rate) * 5,
            }
            return create_message(
                "VISCEROMOTOR_OUTPUT", result, self.config.service_name
            )
        return None

    async def shutdown(self) -> None:
        self.log("info", "Cingulate Autonomic Service shutdown")
