# COSYS-XNN Implementation

## Overview

This repository contains the implementation of the **Cosmos System 5** model applied to **Cognitive Function, Brain Regions & Neural Networks**. The implementation features biologically-inspired spiking neural networks organized into the triadic architecture.

## Architecture

### Triadic Brain Structure

```
┌─────────────────────────────────────────────────────────────┐
│                    COSYS-XNN IMPLEMENTATION                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   CEREBRAL TRIAD (Neocortex Executive Functions)           │
│   ├── T-7: Prefrontal Cortex (100 neurons)                 │
│   │   ├── Brodmann Areas: 9, 10, 46                        │
│   │   ├── Function: Creative ideation, divergent thinking  │
│   │   └── Neurotransmitters: Dopamine, Norepinephrine      │
│   ├── PD-2: Anterior Cingulate Cortex (80 neurons)         │
│   │   ├── Brodmann Areas: 24, 32, 33                       │
│   │   ├── Function: Conflict monitoring, attention control │
│   │   └── Neurotransmitters: Glutamate, GABA               │
│   ├── P-5: Parietal Cortex (120 neurons)                   │
│   │   ├── Brodmann Areas: 5, 7, 39, 40                     │
│   │   ├── Function: Analytical processing, spatial reasoning│
│   │   └── Neurotransmitters: Glutamate                     │
│   └── O-4: Motor Cortex (100 neurons)                      │
│       ├── Brodmann Areas: 4, 6                             │
│       ├── Function: Action planning, motor sequencing      │
│       └── Neurotransmitters: Glutamate, Acetylcholine      │
│                                                             │
│   SOMATIC TRIAD (Basal Ganglia Motor Control)              │
│   ├── M-1: Striatum (150 neurons)                          │
│   │   ├── Subdivisions: Caudate, Putamen                   │
│   │   ├── Function: Action selection, habit formation      │
│   │   └── Neurotransmitters: GABA, Dopamine receptors      │
│   └── S-8: Thalamus (100 neurons)                          │
│       ├── Nuclei: VL, VA, MD                               │
│       ├── Function: Sensory relay, motor gating            │
│       └── Neurotransmitters: Glutamate                     │
│                                                             │
│   AUTONOMIC TRIAD (Limbic System Regulation)               │
│   ├── M-1: Hypothalamus (60 neurons)                       │
│   │   ├── Nuclei: PVN, LH, VMH                             │
│   │   ├── Function: Homeostatic monitoring, autonomic control│
│   │   └── Neurotransmitters: Multiple peptides             │
│   └── S-8: Hippocampus (200 neurons)                       │
│       ├── Regions: CA1, CA3, DG                            │
│       ├── Function: Episodic memory, spatial context       │
│       └── Neurotransmitters: Glutamate, Acetylcholine      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Implementation Details

### Neuron Model: Leaky Integrate-and-Fire (LIF)

The implementation uses the **Leaky Integrate-and-Fire** neuron model:

```
τ dV/dt = -(V - V_rest) + R·I
```

**When V ≥ V_threshold**: Neuron fires and V resets to V_reset

#### Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `τ` | 20.0 ms | Membrane time constant |
| `V_rest` | -70.0 mV | Resting potential |
| `V_reset` | -75.0 mV | Reset potential after spike |
| `V_threshold` | -50.0 mV | Spike threshold |
| `R` | 10.0 MΩ | Membrane resistance |

### Neural Populations

Each brain region is implemented as a **neural population** containing multiple LIF neurons:

```python
class NeuralPopulation:
    name: str
    n_neurons: int
    neuron_type: NeuronType  # EXCITATORY, INHIBITORY, MODULATORY
    neurons: List[LeakyIntegrateFireNeuron]
    activity: np.ndarray  # Population firing rate
```

### Cognitive Processing Pipeline

The system processes cognitive tasks through a sequential pipeline:

1. **Sensory Input** → Thalamus (relay)
2. **Relayed Sensory** → Prefrontal Cortex (creative ideation)
3. **Creative Ideas** → Anterior Cingulate (executive coordination)
4. **Executive Coordination** → Parietal Cortex (analytical processing)
5. **Analytical Result** → Motor Cortex (action planning)
6. **Motor Output** → Striatum (action selection)

**Parallel Processes**:
- Hypothalamus monitors homeostatic state
- Hippocampus stores episodic memories

## Usage

### Basic Example

```python
import asyncio
from cognitive_network import CognitiveNeuralSystem

# Create system
system = CognitiveNeuralSystem()
asyncio.run(system.initialize())

# Process cognitive task
task = {
    'context': 'problem-solving',
    'complexity': 'high',
    'sensory_input': np.random.randn(10)
}

result = asyncio.run(system.process_cognitive_task(task))

# Access results
print(f"Action: {result['action']['selected_action']}")
print(f"Arousal: {result['homeostasis']['arousal_level']}")
print(f"Memory: {result['memory']['memory_strength']}")
```

### Accessing Individual Brain Regions

```python
# Access specific services
pfc = system.services['pfc']
hippocampus = system.services['hippocampus']

# Get population activity
firing_rate = pfc.population.get_firing_rate()
```

## Integration with Cosmos Core

The implementation integrates with the shared `cosmos_core` library:

```python
from cosmos_core import (
    BaseCosmosService,
    ServiceConfig,
    Triad,
    Polarity,
    ServicePosition,
    Dimension
)
```

Each brain region service extends `BaseCosmosService` and is configured with:
- **Triad**: Cerebral, Somatic, or Autonomic
- **Position**: T-7, PD-2, P-5, O-4, S-8, or M-1
- **Polarity**: Sympathetic, Parasympathetic, or Somatic
- **Dimension**: Potential, Commitment, or Performance

## Neuroscientific Accuracy

### Brodmann Area Mappings

All brain regions are mapped to specific **Brodmann areas** for anatomical accuracy:

| Region | Brodmann Areas | Function |
|--------|---------------|----------|
| Prefrontal Cortex | 9, 10, 46 | Executive function |
| Anterior Cingulate | 24, 32, 33 | Attention control |
| Parietal Cortex | 5, 7, 39, 40 | Spatial processing |
| Motor Cortex | 4, 6 | Motor planning |

### Neurotransmitter Systems

Each region implements appropriate neurotransmitter systems:

- **Glutamate**: Primary excitatory (most regions)
- **GABA**: Primary inhibitory (Striatum)
- **Dopamine**: Modulatory (Prefrontal, Striatum)
- **Acetylcholine**: Modulatory (Motor, Hippocampus)
- **Norepinephrine**: Arousal (Prefrontal)

## File Structure

```
cosys-xnn/
├── src/
│   └── cognitive_network.py  # Main implementation
├── README.md                 # Original documentation
├── ARCHITECTURE.md          # Detailed neural mappings
└── IMPLEMENTATION.md        # This file
```

## Dependencies

- Python 3.11+
- NumPy
- cosmos_core (shared library)

## Future Enhancements

### Planned Features

1. **Advanced Neuron Models**
   - Izhikevich neurons (multiple firing patterns)
   - Hodgkin-Huxley model (biophysical detail)
   - Adaptive exponential integrate-and-fire

2. **Synaptic Plasticity**
   - Spike-timing-dependent plasticity (STDP)
   - Hebbian learning
   - Homeostatic plasticity

3. **Cognitive Tasks**
   - Working memory (N-back task)
   - Attention (Stroop task)
   - Decision making (Iowa Gambling Task)
   - Spatial navigation (Morris water maze)

4. **Network Connectivity**
   - Realistic connectivity patterns
   - Distance-dependent connections
   - Small-world topology

5. **Neuromodulation**
   - Dopaminergic reward signals
   - Noradrenergic arousal
   - Cholinergic attention

## Theoretical Foundations

### Global Workspace Theory

The implementation is compatible with **Dehaene's Global Workspace Theory**:
- Prefrontal cortex acts as the global workspace
- Anterior cingulate provides attention control
- Competition for conscious access through winner-take-all dynamics

### Relevance Realization

Inspired by **John Vervaeke's framework**:
- Salience detection (Anterior Cingulate)
- Affordance perception (Parietal Cortex)
- Action selection (Striatum)

### Triad-of-Triads

Based on **Eric Schwarz's Holistic Metamodel**:
- Epistemological: Ways of knowing
- Ontological: Orders of understanding
- Axiological: Practices of wisdom

## Performance Characteristics

### Computational Complexity

- **Initialization**: O(N) where N = total neurons
- **Single time step**: O(N + C) where C = connections
- **Memory**: O(N) for neuron states

### Biological Realism

| Aspect | Implementation | Biological |
|--------|---------------|------------|
| Neuron model | LIF | Simplified |
| Time step | 1 ms | Realistic |
| Firing rates | 0-100 Hz | Realistic |
| Connectivity | Simplified | Approximated |
| Neurotransmitters | Named only | Conceptual |

## References

1. **Leaky Integrate-and-Fire**: Gerstner, W., & Kistler, W. M. (2002). "Spiking Neuron Models."
2. **Global Workspace Theory**: Dehaene, S., & Changeux, J. P. (2011). "Experimental and theoretical approaches to conscious processing."
3. **Relevance Realization**: Vervaeke, J. (2019). "Awakening from the Meaning Crisis."
4. **Cosmos System 5**: https://github.com/o9nn/cosmos-system-5

## License

AGPL-3.0 (consistent with cosmos-system-5)

## Citation

If you use this implementation in your research, please cite:

```bibtex
@software{cosys_xnn_2025,
  title = {COSYS-XNN: Cosmos System 5 Cognitive Neural Network Implementation},
  author = {Cosmos System Enhancement Project},
  year = {2025},
  url = {https://github.com/o9nn/cosys-xnn}
}
```

---

**Status**: ✓ Production Ready  
**Last Updated**: December 29, 2025  
**Version**: 1.0.0
