"""Autognosis: hierarchical self-awareness and self-optimization."""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class SelfImage:
    level: str
    metrics: Dict[str, float]
    confidence: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class MetaInsight:
    insight: str
    confidence: float
    source_images: List[str]


class SelfMonitor:
    """Observe neural states and detect activation patterns."""

    def observe(self, service_states: Dict[str, float]) -> Dict[str, Any]:
        mean_act = float(np.mean(list(service_states.values()))) if service_states else 0.0
        anomalies = {k: v for k, v in service_states.items() if abs(v - mean_act) > 0.3}
        return {
            "mean_activation": mean_act,
            "anomalies": anomalies,
            "n_services": len(service_states),
        }


class SelfModeler:
    """Build hierarchical self-images from observations."""

    LEVELS = ["neural", "functional", "cognitive", "metacognitive"]

    def build_images(self, observation: Dict[str, Any]) -> List[SelfImage]:
        images = []
        for i, level in enumerate(self.LEVELS):
            confidence = max(
                0.1,
                1.0 - i * 0.1 - len(observation.get("anomalies", {})) * 0.05,
            )
            metrics = {
                "mean_activation": observation.get("mean_activation", 0.0),
                "anomaly_count": len(observation.get("anomalies", {})),
                "level_weight": 1.0 / (i + 1),
            }
            images.append(SelfImage(level=level, metrics=metrics, confidence=confidence))
        return images


class MetaCognition:
    """Generate insights from self-images."""

    def generate_insights(self, images: List[SelfImage]) -> List[MetaInsight]:
        insights = []
        low_conf = [img for img in images if img.confidence < 0.5]
        if low_conf:
            insights.append(
                MetaInsight(
                    "Low-confidence self-models detected — increase monitoring",
                    confidence=0.8,
                    source_images=[img.level for img in low_conf],
                )
            )
        high_anom = [
            img for img in images if img.metrics.get("anomaly_count", 0) > 2
        ]
        if high_anom:
            insights.append(
                MetaInsight(
                    "Multiple neural anomalies — consider homeostatic correction",
                    confidence=0.7,
                    source_images=[img.level for img in high_anom],
                )
            )
        return insights


class SelfOptimizer:
    """Discover and apply adaptive improvements."""

    def optimize(
        self,
        insights: List[MetaInsight],
        service_states: Dict[str, float],
    ) -> Dict[str, Any]:
        actions = []
        for insight in insights:
            if "homeostatic" in insight.insight.lower():
                actions.append(
                    {"type": "homeostatic_scaling", "target": "all", "strength": 0.1}
                )
            if "monitoring" in insight.insight.lower():
                actions.append(
                    {"type": "increase_monitoring", "interval_reduction": 0.5}
                )
        return {"recommended_actions": actions, "n_insights": len(insights)}


class AutognosisOrchestrator:
    """Coordinates the four self-awareness layers."""

    def __init__(self):
        self.monitor = SelfMonitor()
        self.modeler = SelfModeler()
        self.meta = MetaCognition()
        self.optimizer = SelfOptimizer()
        self.current_self_images: List[SelfImage] = []

    async def run_autognosis_cycle(
        self, service_states: Dict[str, float]
    ) -> Dict[str, Any]:
        observation = self.monitor.observe(service_states)
        self.current_self_images = self.modeler.build_images(observation)
        insights = self.meta.generate_insights(self.current_self_images)
        optimization = self.optimizer.optimize(insights, service_states)
        return {
            "observation": observation,
            "self_images": [
                {"level": si.level, "confidence": si.confidence}
                for si in self.current_self_images
            ],
            "insights": [
                {"insight": i.insight, "confidence": i.confidence}
                for i in insights
            ],
            "optimization": optimization,
        }
