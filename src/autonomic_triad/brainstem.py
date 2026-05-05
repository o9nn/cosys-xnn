"""T-7: Brainstem (PAG/LC/NTS) — autonomic reflex triggers, arousal modulation."""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from typing import Optional

from cosmos_core import BaseCosmosService, ServiceConfig, ServiceMessage, create_message
from neural_models.lif_neuron import NeuralPopulation, NeuronType


class BrainstemService(BaseCosmosService):
    """T-7: Brainstem nuclei. PAG for defense, LC for norepinephrine, NTS for visceral."""

    def __init__(self, config: ServiceConfig, n_neurons: int = 60):
        super().__init__(config)
        self.pag = NeuralPopulation("PAG", n_neurons // 3, NeuronType.EXCITATORY)
        self.lc = NeuralPopulation("LC", n_neurons // 3, NeuronType.MODULATORY)
        self.nts = NeuralPopulation("NTS", n_neurons // 3, NeuronType.EXCITATORY)
        self.ne_level = 0.3

    async def initialize(self) -> None:
        self.log("info", "Brainstem Service initialized")
        self.initialized = True

    async def process(self, message: ServiceMessage) -> Optional[ServiceMessage]:
        if message.type in ("EMOTIONAL_STATE", "HOMEOSTATIC_STATE", "SENSORY_INPUT"):
            pag_c = np.random.randn(self.pag.n_neurons) * 3.0
            lc_c = np.random.randn(self.lc.n_neurons) * 3.5
            nts_c = np.random.randn(self.nts.n_neurons) * 2.5
            self.pag.step(pag_c)
            self.lc.step(lc_c)
            self.nts.step(nts_c)
            lc_rate = self.lc.get_firing_rate()
            self.ne_level = 0.9 * self.ne_level + 0.1 * lc_rate
            result = {
                "norepinephrine": self.ne_level,
                "serotonin": max(0.0, 1.0 - self.ne_level),
                "pag_activity": self.pag.get_firing_rate(),
                "lc_activity": lc_rate,
                "nts_activity": self.nts.get_firing_rate(),
                "arousal_signal": self.ne_level * 10,
                "autonomic_mode": (
                    "sympathetic" if self.ne_level > 0.5 else "parasympathetic"
                ),
            }
            return create_message("AROUSAL_SIGNAL", result, self.config.service_name)
        return None

    async def shutdown(self) -> None:
        self.log("info", "Brainstem Service shutdown")
