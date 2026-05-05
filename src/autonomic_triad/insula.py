"""P-5: Insula (anterior/posterior) — interoceptive processing, emotional awareness."""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from typing import Optional

from cosmos_core import BaseCosmosService, ServiceConfig, ServiceMessage, create_message
from neural_models.lif_neuron import NeuralPopulation, NeuronType


class InsulaService(BaseCosmosService):
    """P-5: Insula. Anterior for emotional awareness, posterior for interoception."""

    def __init__(self, config: ServiceConfig, n_neurons: int = 80):
        super().__init__(config)
        self.anterior = NeuralPopulation(
            "Anterior Insula", n_neurons // 2, NeuronType.EXCITATORY
        )
        self.posterior = NeuralPopulation(
            "Posterior Insula", n_neurons // 2, NeuronType.EXCITATORY
        )

    async def initialize(self) -> None:
        self.log("info", "Insula Service initialized")
        self.initialized = True

    async def process(self, message: ServiceMessage) -> Optional[ServiceMessage]:
        if message.type in ("EMOTIONAL_STATE", "AROUSAL_SIGNAL", "HOMEOSTATIC_STATE"):
            ant_c = np.random.randn(self.anterior.n_neurons) * 3.5
            pos_c = np.random.randn(self.posterior.n_neurons) * 3.0
            self.anterior.step(ant_c)
            self.posterior.step(pos_c)
            ant_rate = self.anterior.get_firing_rate()
            pos_rate = self.posterior.get_firing_rate()
            result = {
                "emotional_awareness": ant_rate * 10,
                "interoceptive_signal": pos_rate * 10,
                "body_state": (ant_rate + pos_rate) / 2,
                "anterior_activity": ant_rate,
                "posterior_activity": pos_rate,
            }
            return create_message(
                "INTEROCEPTIVE_STATE", result, self.config.service_name
            )
        return None

    async def shutdown(self) -> None:
        self.log("info", "Insula Service shutdown")
