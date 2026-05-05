"""
Relevance Realization Engine — Vervaeke's framework.
Continuous balancing of cognitive trade-offs for adaptive salience.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class SalienceFeature:
    name: str
    score: float
    source: str


@dataclass
class Affordance:
    action: str
    feasibility: float
    utility: float


@dataclass
class CognitiveTradeoffs:
    exploration_exploitation: float = 0.5
    breadth_depth: float = 0.5
    stability_flexibility: float = 0.5
    speed_accuracy: float = 0.5
    certainty_openness: float = 0.5


@dataclass
class RelevanceMap:
    salience_features: List[SalienceFeature]
    affordances: List[Affordance]
    attention_allocation: Dict[str, float]
    tradeoffs: CognitiveTradeoffs


class SalienceLandscape:
    def compute(self, context: Dict[str, Any]) -> List[SalienceFeature]:
        features = []
        for key, value in context.items():
            try:
                score = float(abs(hash(str(value)) % 100)) / 100.0
            except Exception:
                score = 0.5
            features.append(SalienceFeature(name=key, score=score, source="context"))
        return sorted(features, key=lambda f: f.score, reverse=True)


class AffordanceSpace:
    _DEFAULT_ACTIONS = [
        "approach", "avoid", "attend", "ignore", "manipulate", "communicate"
    ]

    def detect(
        self, context: Dict[str, Any], salience: List[SalienceFeature]
    ) -> List[Affordance]:
        affordances = []
        top_score = salience[0].score if salience else 0.5
        for action in self._DEFAULT_ACTIONS:
            feasibility = np.random.uniform(0.3, 1.0)
            utility = top_score * feasibility
            affordances.append(
                Affordance(action=action, feasibility=feasibility, utility=utility)
            )
        return sorted(affordances, key=lambda a: a.utility, reverse=True)


class AttentionController:
    def __init__(self):
        self.weights: Dict[str, float] = {}

    def allocate(
        self,
        salience: List[SalienceFeature],
        affordances: List[Affordance],
        exploration_weight: float = 0.5,
    ) -> Dict[str, float]:
        allocation = {}
        for feat in salience[:5]:
            allocation[feat.name] = feat.score * (1.0 - exploration_weight)
        for aff in affordances[:3]:
            allocation[aff.action] = aff.utility * exploration_weight
        total = sum(allocation.values()) or 1.0
        return {k: v / total for k, v in allocation.items()}


class RelevanceRealizationEngine:
    def __init__(self):
        self.salience_landscape = SalienceLandscape()
        self.affordance_space = AffordanceSpace()
        self.attention_controller = AttentionController()
        self._exploration_weight = 0.5

    def realize_relevance(self, context: Dict[str, Any]) -> RelevanceMap:
        salience = self.salience_landscape.compute(context)
        affordances = self.affordance_space.detect(context, salience)
        attention = self.attention_controller.allocate(
            salience, affordances, self._exploration_weight
        )
        tradeoffs = CognitiveTradeoffs(
            exploration_exploitation=self._exploration_weight,
        )
        return RelevanceMap(salience, affordances, attention, tradeoffs)

    def update_exploration(self, reward: float) -> None:
        """Reduce exploration when rewarded (exploitation)."""
        self._exploration_weight = max(
            0.1, min(0.9, self._exploration_weight - 0.01 * reward)
        )
