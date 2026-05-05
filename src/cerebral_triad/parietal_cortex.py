"""P-5: Parietal Cortex — analytical processing, spatial reasoning."""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from typing import Optional

from cosmos_core import BaseCosmosService, ServiceConfig, ServiceMessage, create_message
from neural_models.lif_neuron import NeuralPopulation, NeuronType


class ParietalCortexService(BaseCosmosService):
    """P-5: Brodmann Areas 5, 7, 39, 40. Glutamate."""

    def __init__(self, config: ServiceConfig, n_neurons: int = 120):
        super().__init__(config)
        self.population = NeuralPopulation("Parietal Cortex", n_neurons, NeuronType.EXCITATORY)

    async def initialize(self) -> None:
        self.log("info", "Parietal Cortex Service initialized")
        self.initialized = True

    async def process(self, message: ServiceMessage) -> Optional[ServiceMessage]:
        if message.type == "EXECUTIVE_COORDINATION":
            coordination = message.payload
            input_currents = np.random.randn(self.population.n_neurons) * 4.0
            spikes = self.population.step(input_currents)
            firing_rate = self.population.get_firing_rate()
            analysis = {
                "analytical_depth": firing_rate * 10,
                "spatial_reasoning": firing_rate * 8,
                "processed_coordination": coordination,
                "ready_for_output": firing_rate > 0.3,
            }
            return create_message(
                "ANALYTICAL_RESULT",
                analysis,
                self.config.service_name,
                "cerebral:O-4",
            )
        return None

    async def shutdown(self) -> None:
        self.log("info", "Parietal Cortex Service shutdown")
