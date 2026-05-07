"""Holistic Metamodel: integrates Ennead as runtime constraints on service behavior."""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "models"))

from ennead_cognitive import EnneadCognitiveModel

from typing import Any, Dict, Optional


class HolisticMetamodel:
    """
    Maps all 18 services to their epistemic/ontological/axiological roles
    and applies Ennead constraints to gate or modulate service behavior.
    """

    SERVICE_ROLES: Dict[str, Dict[str, str]] = {
        "pfc":              {"epistemic": "Propositional/Nomological", "triad": "Cerebral",  "position": "T-7"},
        "acc":              {"epistemic": "Normative",                  "triad": "Cerebral",  "position": "PD-2"},
        "parietal":         {"epistemic": "Perspectival",               "triad": "Cerebral",  "position": "P-5"},
        "motor":            {"epistemic": "Mastery/Morality",           "triad": "Cerebral",  "position": "O-4"},
        "striatum":         {"epistemic": "Procedural",                 "triad": "Somatic",   "position": "M-1"},
        "thalamus":         {"epistemic": "Perspectival (relay)",       "triad": "Somatic",   "position": "S-8"},
        "caudate":          {"epistemic": "Procedural (dev)",           "triad": "Somatic",   "position": "PD-2"},
        "putamen":          {"epistemic": "Procedural (memory)",        "triad": "Somatic",   "position": "T-7"},
        "globus_pallidus":  {"epistemic": "Procedural (gate)",          "triad": "Somatic",   "position": "P-5"},
        "substantia_nigra": {"epistemic": "Meaning (reward)",           "triad": "Somatic",   "position": "O-4"},
        "hypothalamus":     {"epistemic": "Participatory (body)",       "triad": "Autonomic", "position": "M-1"},
        "hippocampus":      {"epistemic": "Narrative/Participatory",    "triad": "Autonomic", "position": "S-8"},
        "amygdala":         {"epistemic": "Normative (emotion)",        "triad": "Autonomic", "position": "PD-2"},
        "brainstem":        {"epistemic": "Participatory (reflex)",     "triad": "Autonomic", "position": "T-7"},
        "insula":           {"epistemic": "Participatory (intero)",     "triad": "Autonomic", "position": "P-5"},
        "cingulate_autonomic": {"epistemic": "Mastery (viscero)",       "triad": "Autonomic", "position": "O-4"},
    }

    def __init__(self):
        self.ennead = EnneadCognitiveModel()

    def get_role(self, service_key: str) -> Optional[Dict[str, str]]:
        return self.SERVICE_ROLES.get(service_key)

    def evaluate_system(self, service_states: Dict[str, float]) -> Dict[str, Any]:
        """
        service_states: {service_key: firing_rate}
        Returns a full Ennead evaluation plus role annotations.
        """
        ennead_eval = self.ennead.evaluate(service_states)
        annotated = {}
        for key, rate in service_states.items():
            role = self.SERVICE_ROLES.get(key, {})
            annotated[key] = {
                "firing_rate": rate,
                "epistemic_role": role.get("epistemic", "unknown"),
                "triad": role.get("triad", "unknown"),
                "position": role.get("position", "unknown"),
            }
        return {"ennead": ennead_eval, "services": annotated}

    def apply_constraints(self, service_key: str, proposed_gain: float) -> float:
        """Returns a possibly-modified gain. Currently a pass-through with clamping."""
        return max(0.0, min(2.0, proposed_gain))
