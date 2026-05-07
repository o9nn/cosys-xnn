"""
Cognitive Task Suite — benchmark tasks for validating the full model.
Each task injects ServiceMessages and collects outputs from relevant services.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from scipy.stats import norm
from typing import Any, Dict

from cosmos_core import create_message


async def n_back_task(system: Any, n: int = 2, sequence_length: int = 20) -> Dict[str, Any]:
    """
    N-back working memory task.
    Engages PFC (T-7), ACC (PD-2), Parietal (P-5) in the Cerebral Triad.
    Returns hit_rate, false_alarm_rate, d_prime.
    """
    items = np.random.randint(0, 9, sequence_length).tolist()
    targets = [i for i in range(n, sequence_length) if items[i] == items[i - n]]
    hits = 0
    false_alarms = 0
    pfc_rates = []
    for t, item in enumerate(items):
        result = await system.process_cognitive_task({
            "context": "n_back",
            "item": int(item),
            "step": t,
            "complexity": "high",
        })
        if result.get("action"):
            response = result["action"].get("selected_action", "inhibit") == "execute"
            is_target = t in targets
            if is_target and response:
                hits += 1
            elif not is_target and response:
                false_alarms += 1
        if "pfc" in system.services:
            pfc_rates.append(system.services["pfc"].population.get_firing_rate())
    hit_rate = hits / max(len(targets), 1)
    fa_rate = false_alarms / max(sequence_length - len(targets), 1)
    d_prime = _d_prime(hit_rate, fa_rate)
    return {
        "task": "n_back",
        "n": n,
        "hit_rate": hit_rate,
        "false_alarm_rate": fa_rate,
        "d_prime": d_prime,
        "mean_pfc_rate": float(np.mean(pfc_rates)) if pfc_rates else 0.0,
    }


async def stroop_task(system: Any, n_trials: int = 20) -> Dict[str, Any]:
    """
    Stroop attention task.
    Conflict monitoring via ACC (PD-2), inhibition via Striatum (M-1).
    """
    congruent_rt = []
    incongruent_rt = []
    acc_rates = []
    for i in range(n_trials):
        congruent = i % 2 == 0
        result = await system.process_cognitive_task({
            "context": "stroop",
            "congruent": congruent,
            "color_word": "RED" if congruent else "BLUE",
            "ink_color": "red",
            "complexity": "medium",
        })
        if result.get("action"):
            action_val = result["action"].get("action_value", 5.0)
            rt = 1.0 / max(action_val, 0.1)
            if congruent:
                congruent_rt.append(rt)
            else:
                incongruent_rt.append(rt)
        if "acc" in system.services:
            acc_rates.append(system.services["acc"].population.get_firing_rate())
    return {
        "task": "stroop",
        "mean_congruent_rt": float(np.mean(congruent_rt)) if congruent_rt else 0.0,
        "mean_incongruent_rt": float(np.mean(incongruent_rt)) if incongruent_rt else 0.0,
        "stroop_effect": (
            float(np.mean(incongruent_rt)) - float(np.mean(congruent_rt))
            if congruent_rt and incongruent_rt
            else 0.0
        ),
        "mean_acc_rate": float(np.mean(acc_rates)) if acc_rates else 0.0,
    }


async def iowa_gambling_task(system: Any, n_trials: int = 20) -> Dict[str, Any]:
    """
    Iowa Gambling Task.
    Reward prediction via Substantia Nigra (O-4), emotional valence via Amygdala (PD-2).
    """
    decks = {
        "A": {"reward": 100, "loss_prob": 0.5, "loss_amount": 250},
        "B": {"reward": 100, "loss_prob": 0.1, "loss_amount": 1250},
        "C": {"reward": 50,  "loss_prob": 0.5, "loss_amount": 50},
        "D": {"reward": 50,  "loss_prob": 0.1, "loss_amount": 250},
    }
    deck_choices = []
    total_gain = 0.0
    sn_rates = []
    for i in range(n_trials):
        deck_key = list(decks.keys())[i % 4]
        d = decks[deck_key]
        reward = d["reward"]
        loss = d["loss_amount"] if np.random.rand() < d["loss_prob"] else 0
        net = reward - loss
        total_gain += net
        result = await system.process_cognitive_task({
            "context": "iowa_gambling",
            "deck": deck_key,
            "reward": reward,
            "loss": loss,
            "net": net,
            "complexity": "high",
        })
        deck_choices.append(deck_key)
        if "substantia_nigra" in system.services:
            sn_rates.append(
                system.services["substantia_nigra"].population.get_firing_rate()
            )
    advantageous = sum(1 for d in deck_choices if d in ("C", "D")) / max(n_trials, 1)
    return {
        "task": "iowa_gambling",
        "total_gain": total_gain,
        "advantageous_rate": advantageous,
        "deck_distribution": {d: deck_choices.count(d) for d in "ABCD"},
        "mean_sn_rate": float(np.mean(sn_rates)) if sn_rates else 0.0,
    }


async def morris_maze_task(system: Any, n_trials: int = 10) -> Dict[str, Any]:
    """
    Morris Water Maze (spatial navigation).
    Hippocampal place cell encoding via Hippocampus (S-8).
    """
    latencies = []
    hipp_rates = []
    for i in range(n_trials):
        position = np.random.randn(2).tolist()
        result = await system.process_cognitive_task({
            "context": "spatial_navigation",
            "position": position,
            "trial": i,
            "complexity": "medium",
        })
        memory_strength = 1.0
        if result.get("memory"):
            memory_strength = max(0.1, result["memory"].get("memory_strength", 1.0))
        latency = 10.0 / memory_strength
        latencies.append(latency)
        if "hippocampus" in system.services:
            hipp_rates.append(
                system.services["hippocampus"].population.get_firing_rate()
            )
    return {
        "task": "morris_maze",
        "mean_escape_latency": float(np.mean(latencies)),
        "latency_improvement": float(latencies[0] - latencies[-1]) if len(latencies) > 1 else 0.0,
        "mean_hippocampus_rate": float(np.mean(hipp_rates)) if hipp_rates else 0.0,
    }


def _d_prime(hit_rate: float, fa_rate: float) -> float:
    hr = np.clip(hit_rate, 0.01, 0.99)
    fa = np.clip(fa_rate, 0.01, 0.99)
    return float(norm.ppf(hr) - norm.ppf(fa))
