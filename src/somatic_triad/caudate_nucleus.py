"""PD-2: Caudate Nucleus — motor development, sequence learning."""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from typing import Optional

from cosmos_core import BaseCosmosService, ServiceConfig, ServiceMessage, create_message
from neural_models.lif_neuron import NeuralPopulation, NeuronType


class CaudateNucleusService(BaseCosmosService):
    """PD-2: Caudate Nucleus. Sequences motor programs, part of dorsal striatum."""

    def __init__(self, config: ServiceConfig, n_neurons: int = 80):
        super().__init__(config)
        self.population = NeuralPopulation("Caudate Nucleus", n_neurons, NeuronType.INHIBITORY)
        self.sequence_buffer: list = []

    async def initialize(self) -> None:
        self.log("info", "Caudate Nucleus Service initialized")
        self.initialized = True

    async def process(self, message: ServiceMessage) -> Optional[ServiceMessage]:
        if message.type in ("MOTOR_OUTPUT", "ACTION_SELECTED"):
            input_currents = np.random.randn(self.population.n_neurons) * 3.0
            spikes = self.population.step(input_currents)
            firing_rate = self.population.get_firing_rate()
            self.sequence_buffer.append(firing_rate)
            if len(self.sequence_buffer) > 20:
                self.sequence_buffer.pop(0)
            result = {
                "sequence_length": len(self.sequence_buffer),
                "sequence_coherence": float(np.std(self.sequence_buffer)),
                "development_level": firing_rate * 10,
                "source": message.payload,
            }
            return create_message("MOTOR_SEQUENCE", result, self.config.service_name)
        return None

    async def shutdown(self) -> None:
        self.log("info", "Caudate Nucleus Service shutdown")
