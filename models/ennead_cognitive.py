"""Ennead Meta-System: 9-fold cognitive structure based on Eric Schwarz's Holistic Metamodel."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CognitiveComponent:
    name: str
    neural_substrate: str
    function: str
    triad_role: str
    service_key: Optional[str] = None


# ── Triad I: Ways of Knowing ──────────────────────────────────────────────────
class WaysOfKnowingTriad:
    """Epistemological triad: how we know things."""

    def __init__(self):
        self.propositional = CognitiveComponent(
            "Propositional", "Left PFC (BA 9, 44–47)",
            "Facts, beliefs, declarative knowledge", "T-7", "pfc",
        )
        self.procedural = CognitiveComponent(
            "Procedural", "Basal Ganglia, Cerebellum",
            "Skills, motor programs, implicit knowledge", "M-1/S-8", "striatum",
        )
        self.perspectival = CognitiveComponent(
            "Perspectival", "Parietal Cortex, TPJ",
            "Salience, framing, attentional set", "P-5", "parietal",
        )
        self.participatory = CognitiveComponent(
            "Participatory", "Default Mode Network",
            "Identity, self-model, being-in-the-world", "U3", "hippocampus",
        )

    def components(self) -> List[CognitiveComponent]:
        return [self.propositional, self.procedural, self.perspectival, self.participatory]


# ── Triad II: Orders of Understanding ────────────────────────────────────────
class OrdersOfUnderstandingTriad:
    """Ontological triad: how we understand reality."""

    def __init__(self):
        self.nomological = CognitiveComponent(
            "Nomological", "Dorsolateral PFC",
            "Causal reasoning, law-like understanding", "T-7", "pfc",
        )
        self.normative = CognitiveComponent(
            "Normative", "Ventromedial PFC, OFC",
            "Value judgments, ethical reasoning", "PD-2", "acc",
        )
        self.narrative = CognitiveComponent(
            "Narrative", "Temporal Cortex, Hippocampus",
            "Story construction, temporal coherence", "S-8", "hippocampus",
        )

    def components(self) -> List[CognitiveComponent]:
        return [self.nomological, self.normative, self.narrative]


# ── Triad III: Practices of Wisdom ───────────────────────────────────────────
class PracticesOfWisdomTriad:
    """Axiological triad: how we enact wisdom."""

    def __init__(self):
        self.morality = CognitiveComponent(
            "Morality", "Orbitofrontal Cortex",
            "Virtue, ethical behavior, moral intuition", "O-4", "motor",
        )
        self.meaning = CognitiveComponent(
            "Meaning", "DMN + Salience Networks",
            "Coherence, purpose, relevance realization", "U2/U3", "hippocampus",
        )
        self.mastery = CognitiveComponent(
            "Mastery", "Motor + Cerebellar Circuits",
            "Excellence, expertise, flow states", "O-4", "motor",
        )

    def components(self) -> List[CognitiveComponent]:
        return [self.morality, self.meaning, self.mastery]


# ── Ennead Orchestrator ───────────────────────────────────────────────────────
class EnneadCognitiveModel:
    """Integrates the three triads into the 9-fold meta-system."""

    def __init__(self):
        self.ways_of_knowing = WaysOfKnowingTriad()
        self.orders_of_understanding = OrdersOfUnderstandingTriad()
        self.practices_of_wisdom = PracticesOfWisdomTriad()

    def all_components(self) -> List[CognitiveComponent]:
        return (
            self.ways_of_knowing.components()
            + self.orders_of_understanding.components()
            + self.practices_of_wisdom.components()
        )

    def get_service_roles(self) -> Dict[str, List[str]]:
        """Map service keys → list of epistemic/ontological/axiological roles."""
        roles: Dict[str, List[str]] = {}
        for comp in self.all_components():
            if comp.service_key:
                roles.setdefault(comp.service_key, []).append(comp.name)
        return roles

    def evaluate(self, service_states: Dict[str, Any]) -> Dict[str, Any]:
        """
        Given {service_key: firing_rate}, return an ennead evaluation.
        Higher activity in a service enriches the corresponding cognitive modes.
        """
        evaluation: Dict[str, Any] = {}
        for comp in self.all_components():
            rate = (
                service_states.get(comp.service_key, 0.0)
                if comp.service_key
                else 0.0
            )
            evaluation[comp.name] = {
                "substrate": comp.neural_substrate,
                "function": comp.function,
                "activation": float(rate),
                "active": float(rate) > 0.3,
            }
        return evaluation
