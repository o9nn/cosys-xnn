"""Predictive processing hierarchy implementing Karl Friston's free-energy principle."""

import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class PredictiveLevel:
    """Single level in the predictive processing hierarchy."""
    level_idx: int
    n_units: int = 64
    learning_rate: float = 0.01

    def __post_init__(self):
        self.prediction = np.zeros(self.n_units)
        self.state = np.zeros(self.n_units)
        self._weights = np.random.randn(self.n_units, self.n_units) * 0.01

    def generate_prediction(self) -> np.ndarray:
        return np.tanh(self._weights @ self.state)

    def encode(self, prediction_error: np.ndarray) -> np.ndarray:
        """Encode prediction error for the next level up."""
        self.state = np.tanh(prediction_error)
        return self.state

    def update_prediction(
        self, higher_state: np.ndarray, prediction_error: np.ndarray
    ) -> None:
        """Update internal weights from top-down signal and error."""
        delta = self.learning_rate * np.outer(prediction_error, higher_state)
        self._weights += delta[: self.n_units, : self.n_units]
        self._weights = np.clip(self._weights, -2.0, 2.0)

    def get_state(self) -> np.ndarray:
        return self.state.copy()


class PredictiveProcessingHierarchy:
    """6-level predictive coding hierarchy."""

    def __init__(self, n_levels: int = 6, n_units: int = 64):
        self.levels = [PredictiveLevel(i, n_units) for i in range(n_levels)]

    def process(self, sensory_input: np.ndarray) -> dict:
        """Bottom-up prediction errors meet top-down predictions."""
        n = self.levels[0].n_units
        inp = np.zeros(n)
        inp[: min(len(sensory_input), n)] = sensory_input[:n]

        # Bottom-up pass
        prediction_errors = []
        current = inp
        for level in self.levels:
            pred = level.generate_prediction()
            error = current - pred
            prediction_errors.append(error)
            current = level.encode(error)

        # Top-down pass
        for i in reversed(range(len(self.levels) - 1)):
            self.levels[i].update_prediction(
                self.levels[i + 1].get_state(), prediction_errors[i]
            )

        return {
            "predictions": [l.prediction.copy() for l in self.levels],
            "errors": [e.copy() for e in prediction_errors],
            "total_error": float(sum(np.mean(e ** 2) for e in prediction_errors)),
        }
