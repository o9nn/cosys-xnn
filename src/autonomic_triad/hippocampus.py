"""S-8: Hippocampus (CA1, CA3, DG) — episodic memory, spatial context."""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from typing import Optional

from cosmos_core import BaseCosmosService, ServiceConfig, ServiceMessage, create_message
from neural_models.lif_neuron import NeuralPopulation, NeuronType


class HippocampusService(BaseCosmosService):
    """S-8: Hippocampus CA1/CA3/DG. Glutamate, Acetylcholine."""

    def __init__(self, config: ServiceConfig, n_neurons: int = 200):
        super().__init__(config)
        self.population = NeuralPopulation("Hippocampus", n_neurons, NeuronType.EXCITATORY)
        self.memory_buffer = []
        self.max_memories = 100

    async def initialize(self) -> None:
        self.log("info", "Hippocampus Service initialized")
        self.initialized = True

    async def process(self, message: ServiceMessage) -> Optional[ServiceMessage]:
        input_currents = np.random.randn(self.population.n_neurons) * 3.0
        spikes = self.population.step(input_currents)
        firing_rate = self.population.get_firing_rate()
        self.memory_buffer.append({
            "timestamp": message.timestamp,
            "type": message.type,
            "firing_rate": firing_rate,
        })
        if len(self.memory_buffer) > self.max_memories:
            self.memory_buffer.pop(0)
        memory_state = {
            "memory_strength": firing_rate * 10,
            "context_richness": len(self.memory_buffer),
            "recent_memories": self.memory_buffer[-5:],
        }
        return create_message("MEMORY_STATE", memory_state, self.config.service_name)

    async def shutdown(self) -> None:
        self.log("info", "Hippocampus Service shutdown")
