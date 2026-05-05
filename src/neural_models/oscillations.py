"""Neural oscillations and cross-frequency coupling."""

import numpy as np
from scipy.signal import hilbert

# Canonical frequency band ranges (Hz)
FREQUENCY_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta":  (12.0, 30.0),
    "gamma": (30.0, 100.0),
}


class CrossFrequencyCoupling:
    """Phase-amplitude coupling between low-frequency phase and high-frequency amplitude."""

    def compute_pac(
        self,
        low_freq_signal: np.ndarray,
        high_freq_signal: np.ndarray,
        n_bins: int = 18,
    ) -> float:
        """
        Modulation Index (Tort et al. 2010).
        Returns a value in [0, 1] where 0 = no coupling.
        """
        if len(low_freq_signal) < 4 or len(high_freq_signal) < 4:
            return 0.0
        phase = np.angle(hilbert(low_freq_signal))
        amplitude = np.abs(hilbert(high_freq_signal))
        return self._modulation_index(phase, amplitude, n_bins)

    def _modulation_index(
        self, phase: np.ndarray, amplitude: np.ndarray, n_bins: int
    ) -> float:
        bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
        mean_amp = np.zeros(n_bins)
        for i in range(n_bins):
            mask = (phase >= bin_edges[i]) & (phase < bin_edges[i + 1])
            if mask.sum() > 0:
                mean_amp[i] = amplitude[mask].mean()
        if mean_amp.sum() == 0:
            return 0.0
        p = mean_amp / mean_amp.sum()
        q = np.ones(n_bins) / n_bins
        kl_div = np.sum(p * np.log((p + 1e-10) / q))
        return kl_div / np.log(n_bins)
