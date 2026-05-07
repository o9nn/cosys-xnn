"""M-1: Striatum — action selection, habit formation."""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from typing import Optional

from cosmos_core import BaseCosmosService, ServiceConfig, ServiceMessage, create_message
from neural_models.lif_neuron import NeuralPopulation, NeuronType


class StriatumService(BaseCosmosService):
    """M-1: Striatum (Caudate, Putamen). GABA, Dopamine receptors."""

    def __init__(self, config: ServiceConfig, n_neurons: int = 150):
        super().__init__(config)
        self.population = NeuralPopulation("Striatum", n_neurons, NeuronType.INHIBITORY)
        self.action_values = {}

    async def initialize(self) -> None:
        self.log("info", "Striatum Service initialized")
        self.initialized = True

    async def process(self, message: ServiceMessage) -> Optional[ServiceMessage]:
        if message.type == "MOTOR_OUTPUT":
            motor_plan = message.payload
            input_currents = np.random.randn(self.population.n_neurons) * 3.5
            spikes = self.population.step(input_currents)
            firing_rate = self.population.get_firing_rate()
            action_selection = {
                "selected_action": "execute" if firing_rate > 0.3 else "inhibit",
                "action_value": firing_rate * 10,
                "habit_strength": firing_rate * 5,
                "motor_plan": motor_plan,
            }
            return create_message(
                "ACTION_SELECTED", action_selection, self.config.service_name
            )
        return None

    async def shutdown(self) -> None:
        self.log("info", "Striatum Service shutdown")
