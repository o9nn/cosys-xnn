"""Neural models: LIF neurons, plasticity rules, oscillations, predictive coding."""

from neural_models.lif_neuron import NeuronType, LeakyIntegrateFireNeuron, NeuralPopulation, SynapticMatrix
from neural_models.plasticity import hebbian_update, stdp_update, homeostatic_scaling
from neural_models.oscillations import CrossFrequencyCoupling, FREQUENCY_BANDS
from neural_models.predictive import PredictiveProcessingHierarchy, PredictiveLevel

__all__ = [
    "NeuronType", "LeakyIntegrateFireNeuron", "NeuralPopulation", "SynapticMatrix",
    "hebbian_update", "stdp_update", "homeostatic_scaling",
    "CrossFrequencyCoupling", "FREQUENCY_BANDS",
    "PredictiveProcessingHierarchy", "PredictiveLevel",
]
