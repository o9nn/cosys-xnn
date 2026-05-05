"""Ontogenesis: self-generating, evolving neural kernels."""

import numpy as np
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional


class DevelopmentStage(Enum):
    EMBRYONIC = auto()
    JUVENILE = auto()
    MATURE = auto()
    SENESCENT = auto()


@dataclass
class NeuralGene:
    gene_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    weights: np.ndarray = field(default_factory=lambda: np.random.randn(8))
    expression_level: float = 1.0


@dataclass
class NeuralKernelGenome:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    generation: int = 0
    lineage: List[str] = field(default_factory=list)
    genes: List[NeuralGene] = field(default_factory=list)
    fitness: float = 0.0
    age: int = 0

    def __post_init__(self):
        if not self.genes:
            self.genes = [NeuralGene() for _ in range(4)]

    @property
    def stage(self) -> DevelopmentStage:
        if self.age < 10:
            return DevelopmentStage.EMBRYONIC
        if self.age < 30:
            return DevelopmentStage.JUVENILE
        if self.age < 80:
            return DevelopmentStage.MATURE
        return DevelopmentStage.SENESCENT

    @property
    def plasticity(self) -> float:
        rates = {
            DevelopmentStage.EMBRYONIC: 1.0,
            DevelopmentStage.JUVENILE: 0.7,
            DevelopmentStage.MATURE: 0.3,
            DevelopmentStage.SENESCENT: 0.05,
        }
        return rates[self.stage]


def b_series_expand(genome: NeuralKernelGenome, h: float = 0.1) -> np.ndarray:
    """
    B-series numerical expansion for neural kernel evolution.
    y_{n+1} = y_n + h * Σ b_i * Φ_i(f, y_n)
    Elementary differentials follow the A000081 tree sequence.
    """
    y = np.concatenate([g.weights for g in genome.genes])
    b = np.array([1.0, 0.5, 1.0 / 6.0])
    phi_1 = y
    phi_2 = y + h * 0.5 * phi_1
    phi_3 = y + h * (phi_1 / 6 + phi_2 / 3)
    return y + h * (b[0] * phi_1 + b[1] * phi_2 + b[2] * phi_3)


class OntogenesisEngine:
    """Manages a population of evolving neural kernel genomes."""

    def __init__(self, population_size: int = 10):
        self.population: List[NeuralKernelGenome] = [
            NeuralKernelGenome() for _ in range(population_size)
        ]

    def step(self) -> None:
        """Age all genomes and evolve them one B-series step."""
        for genome in self.population:
            genome.age += 1
            evolved = b_series_expand(genome)
            total = len(evolved)
            per_gene = total // len(genome.genes)
            for i, gene in enumerate(genome.genes):
                chunk = evolved[i * per_gene : (i + 1) * per_gene]
                gene.weights = (
                    (1 - genome.plasticity) * gene.weights
                    + genome.plasticity * chunk[: len(gene.weights)]
                )

    def select_and_reproduce(self) -> None:
        """Tournament selection: top half reproduces, bottom half is replaced."""
        ranked = sorted(self.population, key=lambda g: g.fitness, reverse=True)
        survivors = ranked[: len(ranked) // 2]
        offspring = []
        for parent in survivors:
            child = NeuralKernelGenome(
                generation=parent.generation + 1,
                lineage=parent.lineage + [parent.id],
                genes=[
                    NeuralGene(
                        weights=g.weights + np.random.randn(*g.weights.shape) * 0.05
                    )
                    for g in parent.genes
                ],
            )
            offspring.append(child)
        self.population = survivors + offspring

    def update_fitness(self, genome_id: str, fitness: float) -> None:
        for g in self.population:
            if g.id == genome_id:
                g.fitness = fitness
                return

    def get_best(self) -> Optional[NeuralKernelGenome]:
        return max(self.population, key=lambda g: g.fitness) if self.population else None

    def get_stage_distribution(self) -> Dict[str, int]:
        dist: Dict[str, int] = {}
        for g in self.population:
            key = g.stage.name
            dist[key] = dist.get(key, 0) + 1
        return dist
