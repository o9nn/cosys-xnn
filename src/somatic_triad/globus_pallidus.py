"""P-5: Globus Pallidus (GPi/GPe) — GABAergic motor gating."""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from typing import Optional

from cosmos_core import BaseCosmosService, ServiceConfig, ServiceMessage, create_message
from neural_models.lif_neuron import NeuralPopulation, NeuronType


class GlobusPallidusService(BaseCosmosService):
    """P-5: Globus Pallidus. Primary GABA inhibition/disinhibition of motor circuits."""

    def __init__(self, config: ServiceConfig, n_neurons: int = 70):
        super().__init__(config)
        self.population = NeuralPopulation("Globus Pallidus", n_neurons, NeuronType.INHIBITORY)
        self.inhibition_threshold = 0.4

    async def initialize(self) -> None:
        self.log("info", "Globus Pallidus Service initialized")
        self.initialized = True

    async def process(self, message: ServiceMessage) -> Optional[ServiceMessage]:
        if message.type in ("HABIT_MEMORY", "MOTOR_SEQUENCE", "ACTION_SELECTED"):
            input_currents = np.random.randn(self.population.n_neurons) * 2.5
            spikes = self.population.step(input_currents)
            firing_rate = self.population.get_firing_rate()
            gating = "disinhibit" if firing_rate < self.inhibition_threshold else "inhibit"
            result = {
                "gating_state": gating,
                "inhibition_level": firing_rate * 10,
                "gpi_activity": firing_rate,
                "gpe_activity": max(0.0, 1.0 - firing_rate),
                "source": message.payload,
            }
            return create_message("MOTOR_GATING", result, self.config.service_name)
        return None

    async def shutdown(self) -> None:
        self.log("info", "Globus Pallidus Service shutdown")
