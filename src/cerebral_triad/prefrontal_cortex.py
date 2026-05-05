"""T-7: Right Prefrontal Cortex — creative ideation, divergent thinking."""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from typing import Optional

from cosmos_core import BaseCosmosService, ServiceConfig, ServiceMessage, create_message
from neural_models.lif_neuron import NeuralPopulation, NeuronType


class PrefrontalCortexService(BaseCosmosService):
    """T-7: Brodmann Areas 9, 10, 46. Dopamine, Norepinephrine."""

    def __init__(self, config: ServiceConfig, n_neurons: int = 100):
        super().__init__(config)
        self.population = NeuralPopulation("Prefrontal Cortex", n_neurons, NeuronType.EXCITATORY)
        self.idea_buffer = []

    async def initialize(self) -> None:
        self.log("info", "Prefrontal Cortex Service initialized")
        self.initialized = True

    async def process(self, message: ServiceMessage) -> Optional[ServiceMessage]:
        if message.type in ("SENSORY_INPUT", "RELAYED_SENSORY"):
            sensory_data = message.payload
            input_currents = np.random.randn(self.population.n_neurons) * 5.0
            spikes = self.population.step(input_currents)
            firing_rate = self.population.get_firing_rate()
            ideas = {
                "firing_rate": firing_rate,
                "active_neurons": int(np.sum(spikes)),
                "idea_strength": firing_rate * 10,
                "context": (
                    sensory_data.get("context", "unknown")
                    if isinstance(sensory_data, dict)
                    else "unknown"
                ),
            }
            return create_message(
                "CREATIVE_IDEAS", ideas, self.config.service_name, "cerebral:PD-2"
            )
        return None

    async def shutdown(self) -> None:
        self.log("info", "Prefrontal Cortex Service shutdown")
