"""PD-2: Amygdala (BLA/CeA) — emotional valence, fear/reward circuits."""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from typing import Optional

from cosmos_core import BaseCosmosService, ServiceConfig, ServiceMessage, create_message
from neural_models.lif_neuron import NeuralPopulation, NeuronType


class AmygdalaService(BaseCosmosService):
    """PD-2: Amygdala. BLA for associative learning, CeA for output."""

    def __init__(self, config: ServiceConfig, n_neurons: int = 90):
        super().__init__(config)
        self.bla = NeuralPopulation("BLA", n_neurons // 2, NeuronType.EXCITATORY)
        self.cea = NeuralPopulation("CeA", n_neurons // 2, NeuronType.INHIBITORY)
        self.fear_threshold = 0.6
        self.valence_history: list = []

    async def initialize(self) -> None:
        self.log("info", "Amygdala Service initialized")
        self.initialized = True

    async def process(self, message: ServiceMessage) -> Optional[ServiceMessage]:
        if message.type in (
            "SENSORY_INPUT",
            "RELAYED_SENSORY",
            "HOMEOSTATIC_STATE",
            "DOPAMINE_SIGNAL",
        ):
            bla_currents = np.random.randn(self.bla.n_neurons) * 4.0
            cea_currents = np.random.randn(self.cea.n_neurons) * 3.0
            self.bla.step(bla_currents)
            self.cea.step(cea_currents)
            bla_rate = self.bla.get_firing_rate()
            cea_rate = self.cea.get_firing_rate()
            valence = bla_rate - cea_rate
            self.valence_history.append(valence)
            if len(self.valence_history) > 50:
                self.valence_history.pop(0)
            fear_response = bla_rate > self.fear_threshold
            result = {
                "emotional_valence": valence,
                "fear_response": fear_response,
                "bla_activity": bla_rate,
                "cea_activity": cea_rate,
                "arousal": (bla_rate + cea_rate) / 2,
                "mean_valence": (
                    float(np.mean(self.valence_history)) if self.valence_history else 0.0
                ),
            }
            return create_message("EMOTIONAL_STATE", result, self.config.service_name)
        return None

    async def shutdown(self) -> None:
        self.log("info", "Amygdala Service shutdown")
