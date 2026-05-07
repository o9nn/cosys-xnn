"""Cognitive Core: relevance realization, autognosis, ontogenesis, holistic metamodel."""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cognitive_core.relevance_realization import RelevanceRealizationEngine, RelevanceMap
from cognitive_core.autognosis import AutognosisOrchestrator
from cognitive_core.ontogenesis import OntogenesisEngine, NeuralKernelGenome
from cognitive_core.holistic_metamodel import HolisticMetamodel

__all__ = [
    "RelevanceRealizationEngine", "RelevanceMap",
    "AutognosisOrchestrator",
    "OntogenesisEngine", "NeuralKernelGenome",
    "HolisticMetamodel",
]
