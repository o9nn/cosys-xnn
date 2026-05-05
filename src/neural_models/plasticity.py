"""Synaptic plasticity rules: Hebbian, STDP, homeostatic."""

import numpy as np


def hebbian_update(pre: float, post: float, learning_rate: float = 0.01) -> float:
    """Neurons that fire together, wire together."""
    return learning_rate * pre * post


def stdp_update(
    pre_time: float,
    post_time: float,
    a_plus: float = 0.01,
    a_minus: float = 0.012,
    tau_plus: float = 20.0,
    tau_minus: float = 20.0,
) -> float:
    """
    Spike-timing-dependent plasticity.
    dt > 0 (post after pre)  → LTP
    dt < 0 (post before pre) → LTD
    """
    dt = post_time - pre_time
    if dt > 0:
        return a_plus * np.exp(-dt / tau_plus)
    elif dt < 0:
        return -a_minus * np.exp(dt / tau_minus)
    return 0.0


def homeostatic_scaling(
    current_rate: float,
    target_rate: float,
    scaling_rate: float = 0.001,
) -> float:
    """
    Synaptic scaling to maintain target firing rate.
    Returns a multiplicative gain factor near 1.0.
    """
    error = target_rate - current_rate
    return 1.0 + scaling_rate * error
