"""O-4: Motor Cortex — action planning, motor sequencing."""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from typing import Optional

from cosmos_core import BaseCosmosService, ServiceConfig, ServiceMessage, create_message
from neural_models.lif_neuron import NeuralPopulation, NeuronType


class MotorCortexService(BaseCosmosService):
    """O-4: Brodmann Areas 4, 6. Glutamate, Acetylcholine."""

    def __init__(self, config: ServiceConfig, n_neurons: int = 100):
        super().__init__(config)
        self.population = NeuralPopulation("Motor Cortex", n_neurons, NeuronType.EXCITATORY)

    async def initialize(self) -> None:
        self.log("info", "Motor Cortex Service initialized")
        self.initialized = True

    async def process(self, message: ServiceMessage) -> Optional[ServiceMessage]:
        if message.type == "ANALYTICAL_RESULT":
            analysis = message.payload
            input_currents = np.random.randn(self.population.n_neurons) * 4.5
            spikes = self.population.step(input_currents)
            firing_rate = self.population.get_firing_rate()
            motor_plan = {
                "action_strength": firing_rate * 10,
                "motor_sequence": list(spikes[:10]),
                "execution_ready": firing_rate > 0.4,
                "analysis_basis": analysis,
            }
            return create_message(
                "MOTOR_OUTPUT", motor_plan, self.config.service_name
            )
        return None

    async def shutdown(self) -> None:
        self.log("info", "Motor Cortex Service shutdown")
