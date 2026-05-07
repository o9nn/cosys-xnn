"""T-7: Putamen — motor memory, habit storage."""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from typing import Optional

from cosmos_core import BaseCosmosService, ServiceConfig, ServiceMessage, create_message
from neural_models.lif_neuron import NeuralPopulation, NeuronType


class PutamenService(BaseCosmosService):
    """T-7: Putamen. Stores motor programs and routines via GABA/Dopamine."""

    def __init__(self, config: ServiceConfig, n_neurons: int = 100):
        super().__init__(config)
        self.population = NeuralPopulation("Putamen", n_neurons, NeuronType.INHIBITORY)
        self.habit_memory: dict = {}

    async def initialize(self) -> None:
        self.log("info", "Putamen Service initialized")
        self.initialized = True

    async def process(self, message: ServiceMessage) -> Optional[ServiceMessage]:
        if message.type in ("MOTOR_SEQUENCE", "MOTOR_OUTPUT"):
            input_currents = np.random.randn(self.population.n_neurons) * 3.5
            spikes = self.population.step(input_currents)
            firing_rate = self.population.get_firing_rate()
            key = message.type
            self.habit_memory[key] = firing_rate
            result = {
                "habit_strength": firing_rate * 10,
                "stored_routines": len(self.habit_memory),
                "consolidation_rate": firing_rate,
                "source": message.payload,
            }
            return create_message("HABIT_MEMORY", result, self.config.service_name)
        return None

    async def shutdown(self) -> None:
        self.log("info", "Putamen Service shutdown")
