"""Somatic Triad: basal ganglia motor control services."""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from somatic_triad.striatum import StriatumService
from somatic_triad.thalamus import ThalamusService
from somatic_triad.caudate_nucleus import CaudateNucleusService
from somatic_triad.putamen import PutamenService
from somatic_triad.globus_pallidus import GlobusPallidusService
from somatic_triad.substantia_nigra import SubstantiaNigraService

__all__ = [
    "StriatumService",
    "ThalamusService",
    "CaudateNucleusService",
    "PutamenService",
    "GlobusPallidusService",
    "SubstantiaNigraService",
]
