"""PD-2: Anterior Cingulate Cortex — conflict monitoring, executive coordination."""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from typing import Optional

from cosmos_core import BaseCosmosService, ServiceConfig, ServiceMessage, create_message
from neural_models.lif_neuron import NeuralPopulation, NeuronType


class AnteriorCingulateService(BaseCosmosService):
    """PD-2: Brodmann Areas 24, 32, 33. Glutamate, GABA."""

    def __init__(self, config: ServiceConfig, n_neurons: int = 80):
        super().__init__(config)
        self.population = NeuralPopulation("Anterior Cingulate", n_neurons, NeuronType.EXCITATORY)
        self.conflict_threshold = 0.5

    async def initialize(self) -> None:
        self.log("info", "Anterior Cingulate Service initialized")
        self.initialized = True

    async def process(self, message: ServiceMessage) -> Optional[ServiceMessage]:
        if message.type == "CREATIVE_IDEAS":
            ideas = message.payload
            input_currents = np.random.randn(self.population.n_neurons) * 3.0
            spikes = self.population.step(input_currents)
            firing_rate = self.population.get_firing_rate()
            conflict_detected = firing_rate > self.conflict_threshold
            coordination = {
                "conflict_level": firing_rate,
                "attention_allocated": firing_rate * 100,
                "executive_control": "high" if conflict_detected else "low",
                "ideas_processed": ideas,
            }
            return create_message(
                "EXECUTIVE_COORDINATION",
                coordination,
                self.config.service_name,
                "cerebral:P-5",
            )
        return None

    async def shutdown(self) -> None:
        self.log("info", "Anterior Cingulate Service shutdown")
