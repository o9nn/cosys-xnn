"""Polarity-Neural Mapping: 18-service polarity model."""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from dataclasses import dataclass
from typing import Dict

try:
    from cosmos_core import Polarity
except ImportError:
    from enum import Enum

    class Polarity(Enum):  # type: ignore[no-redef]
        SYMPATHETIC = "sympathetic"
        PARASYMPATHETIC = "parasympathetic"
        SOMATIC = "somatic"

# Frequency band descriptions
POLARITY_BANDS = {
    Polarity.SYMPATHETIC:     {"band": "gamma",      "range_hz": (30, 100), "pattern": "excitatory"},
    Polarity.PARASYMPATHETIC: {"band": "theta/alpha", "range_hz": (4, 12),  "pattern": "inhibitory"},
    Polarity.SOMATIC:         {"band": "beta",        "range_hz": (12, 30), "pattern": "motor"},
}


@dataclass
class PolarityActivation:
    polarity: Polarity
    band: str
    gain: float
    oscillation_frequency: float
    description: str


class PolarityNeuralMapper:
    """Computes per-service activation patterns from polarity and applies gain modulation."""

    def compute_activation(
        self, polarity: Polarity, base_rate: float
    ) -> PolarityActivation:
        info = POLARITY_BANDS[polarity]
        low, high = info["range_hz"]
        freq = (low + high) / 2
        if polarity == Polarity.SYMPATHETIC:
            gain = 1.0 + base_rate
            desc = "High-frequency gamma; glutamatergic excitation"
        elif polarity == Polarity.PARASYMPATHETIC:
            gain = max(0.1, 1.0 - base_rate * 0.5)
            desc = "Theta/alpha synchronization; GABAergic inhibition"
        else:
            gain = 0.8 + 0.4 * base_rate
            desc = "Beta oscillations; mu-suppression during motor execution"
        return PolarityActivation(
            polarity=polarity,
            band=info["band"],
            gain=gain,
            oscillation_frequency=freq,
            description=desc,
        )

    def modulate_input(
        self, polarity: Polarity, base_current: np.ndarray
    ) -> np.ndarray:
        """Apply polarity-based gain to neural population input currents."""
        base_rate = float(np.mean(np.abs(base_current)))
        activation = self.compute_activation(polarity, base_rate)
        return base_current * activation.gain

    def get_all_service_patterns(
        self, service_polarities: Dict[str, str]
    ) -> Dict[str, PolarityActivation]:
        """
        service_polarities: {service_name: polarity_value_string}
        Returns activation pattern for each service.
        """
        results = {}
        for name, pol_str in service_polarities.items():
            polarity = Polarity(pol_str)
            results[name] = self.compute_activation(polarity, 0.3)
        return results
