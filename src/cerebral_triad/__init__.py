"""Cerebral Triad: neocortex executive services."""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cerebral_triad.prefrontal_cortex import PrefrontalCortexService
from cerebral_triad.anterior_cingulate import AnteriorCingulateService
from cerebral_triad.parietal_cortex import ParietalCortexService
from cerebral_triad.motor_cortex import MotorCortexService

__all__ = [
    "PrefrontalCortexService",
    "AnteriorCingulateService",
    "ParietalCortexService",
    "MotorCortexService",
]
