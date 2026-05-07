"""S-8: Thalamus (VL, VA, MD nuclei) — sensory relay, motor gating."""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from typing import Optional

from cosmos_core import BaseCosmosService, ServiceConfig, ServiceMessage, create_message
from neural_models.lif_neuron import NeuralPopulation, NeuronType


class ThalamusService(BaseCosmosService):
    """S-8: Thalamus nuclei. Glutamate."""

    def __init__(self, config: ServiceConfig, n_neurons: int = 100):
        super().__init__(config)
        self.population = NeuralPopulation("Thalamus", n_neurons, NeuronType.EXCITATORY)

    async def initialize(self) -> None:
        self.log("info", "Thalamus Service initialized")
        self.initialized = True

    async def process(self, message: ServiceMessage) -> Optional[ServiceMessage]:
        if message.type == "SENSORY_INPUT":
            sensory_data = message.payload
            input_currents = np.random.randn(self.population.n_neurons) * 4.0
            spikes = self.population.step(input_currents)
            firing_rate = self.population.get_firing_rate()
            relayed_info = {
                "relay_strength": firing_rate * 10,
                "gating_active": firing_rate > 0.4,
                "sensory_data": sensory_data,
            }
            return create_message(
                "RELAYED_SENSORY",
                relayed_info,
                self.config.service_name,
                "cerebral:T-7",
            )
        return None

    async def shutdown(self) -> None:
        self.log("info", "Thalamus Service shutdown")
