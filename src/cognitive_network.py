"""
COSYS-XNN: Cognitive Neural Network Implementation
===================================================

Cosmos System 5 model applied to cognitive function, brain regions, and neural networks.

This module implements the triadic cognitive architecture:
- Cerebral Triad: Neocortex executive functions (prefrontal, parietal, motor cortex)
- Somatic Triad: Basal ganglia motor control (striatum, thalamus, globus pallidus)
- Autonomic Triad: Limbic system regulation (hypothalamus, hippocampus, amygdala)

Author: Cosmos System Enhancement Project
Date: December 29, 2025
License: AGPL-3.0
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import sys
sys.path.append('/home/ubuntu/cosys-enhancement/shared-cosmos-lib')
from cosmos_core import (
    BaseCosmosService, ServiceConfig, ServiceMessage,
    Triad, Polarity, ServicePosition, Dimension,
    TriadicCoordinator, create_message, setup_logging
)


# ============================================================================
# NEURAL MODELS
# ============================================================================

class NeuronType(Enum):
    """Types of neurons with different dynamics."""
    EXCITATORY = "excitatory"      # Glutamatergic
    INHIBITORY = "inhibitory"      # GABAergic
    MODULATORY = "modulatory"      # Dopaminergic, etc.


@dataclass
class LeakyIntegrateFireNeuron:
    """
    Leaky Integrate-and-Fire (LIF) neuron model.
    
    Membrane potential dynamics:
    τ dV/dt = -(V - V_rest) + R·I
    
    When V ≥ V_threshold, neuron fires and V resets to V_reset.
    """
    # Parameters
    tau: float = 20.0           # Membrane time constant (ms)
    v_rest: float = -70.0       # Resting potential (mV)
    v_reset: float = -75.0      # Reset potential (mV)
    v_threshold: float = -50.0  # Spike threshold (mV)
    resistance: float = 10.0    # Membrane resistance (MΩ)
    
    # State
    v: float = field(default=-70.0)  # Current membrane potential
    spike: bool = field(default=False)
    
    def step(self, input_current: float, dt: float = 1.0) -> bool:
        """
        Update neuron state for one time step.
        
        Args:
            input_current: Input current (nA)
            dt: Time step (ms)
        
        Returns:
            True if neuron spiked, False otherwise
        """
        # Euler integration
        dv = (-(self.v - self.v_rest) + self.resistance * input_current) / self.tau
        self.v += dv * dt
        
        # Check for spike
        if self.v >= self.v_threshold:
            self.v = self.v_reset
            self.spike = True
            return True
        
        self.spike = False
        return False


@dataclass
class NeuralPopulation:
    """
    Population of neurons representing a brain region.
    """
    name: str
    n_neurons: int
    neuron_type: NeuronType
    neurons: List[LeakyIntegrateFireNeuron] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize neurons."""
        self.neurons = [LeakyIntegrateFireNeuron() for _ in range(self.n_neurons)]
        self.activity = np.zeros(self.n_neurons)
    
    def step(self, input_currents: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """
        Update all neurons in the population.
        
        Args:
            input_currents: (n_neurons,) array of input currents
            dt: Time step (ms)
        
        Returns:
            (n_neurons,) array of spike indicators (0 or 1)
        """
        spikes = np.zeros(self.n_neurons)
        for i, neuron in enumerate(self.neurons):
            spikes[i] = float(neuron.step(input_currents[i], dt))
        
        # Update population activity (exponential moving average)
        self.activity = 0.9 * self.activity + 0.1 * spikes
        return spikes
    
    def get_firing_rate(self) -> float:
        """Get population firing rate."""
        return np.mean(self.activity)


# ============================================================================
# CEREBRAL TRIAD: NEOCORTEX EXECUTIVE FUNCTIONS
# ============================================================================

class PrefrontalCortexService(BaseCosmosService):
    """
    T-7: Right Prefrontal Cortex
    Creative ideation, divergent thinking, pattern recognition.
    
    Brodmann Areas: 9, 10, 46
    Neurotransmitters: Dopamine, Norepinephrine
    """
    
    def __init__(self, config: ServiceConfig, n_neurons: int = 100):
        super().__init__(config)
        self.population = NeuralPopulation(
            "Prefrontal Cortex",
            n_neurons,
            NeuronType.EXCITATORY
        )
        self.idea_buffer = []
        
    async def initialize(self) -> None:
        self.log('info', 'Prefrontal Cortex Service initialized')
        self.initialized = True
        
    async def process(self, message: ServiceMessage) -> Optional[ServiceMessage]:
        if message.type in ['SENSORY_INPUT', 'RELAYED_SENSORY']:
            # Generate creative ideas from sensory input
            sensory_data = message.payload
            
            # Simulate neural activity
            input_currents = np.random.randn(self.population.n_neurons) * 5.0
            spikes = self.population.step(input_currents)
            
            # Generate ideas based on firing rate
            firing_rate = self.population.get_firing_rate()
            
            ideas = {
                'firing_rate': firing_rate,
                'active_neurons': int(np.sum(spikes)),
                'idea_strength': firing_rate * 10,
                'context': sensory_data.get('context', 'unknown')
            }
            
            return create_message(
                'CREATIVE_IDEAS',
                ideas,
                self.config.service_name,
                'cerebral:PD-2'
            )
        return None
    
    async def shutdown(self) -> None:
        self.log('info', 'Prefrontal Cortex Service shutdown')


class AnteriorCingulateService(BaseCosmosService):
    """
    PD-2: Anterior Cingulate Cortex
    Conflict monitoring, attention control, executive coordination.
    
    Brodmann Areas: 24, 32, 33
    Neurotransmitters: Glutamate, GABA
    """
    
    def __init__(self, config: ServiceConfig, n_neurons: int = 80):
        super().__init__(config)
        self.population = NeuralPopulation(
            "Anterior Cingulate",
            n_neurons,
            NeuronType.EXCITATORY
        )
        self.conflict_threshold = 0.5
        
    async def initialize(self) -> None:
        self.log('info', 'Anterior Cingulate Service initialized')
        self.initialized = True
        
    async def process(self, message: ServiceMessage) -> Optional[ServiceMessage]:
        if message.type == 'CREATIVE_IDEAS':
            ideas = message.payload
            
            # Simulate conflict detection
            input_currents = np.random.randn(self.population.n_neurons) * 3.0
            spikes = self.population.step(input_currents)
            
            firing_rate = self.population.get_firing_rate()
            conflict_detected = firing_rate > self.conflict_threshold
            
            coordination = {
                'conflict_level': firing_rate,
                'attention_allocated': firing_rate * 100,
                'executive_control': 'high' if conflict_detected else 'low',
                'ideas_processed': ideas
            }
            
            return create_message(
                'EXECUTIVE_COORDINATION',
                coordination,
                self.config.service_name,
                'cerebral:P-5'
            )
        return None
    
    async def shutdown(self) -> None:
        self.log('info', 'Anterior Cingulate Service shutdown')


class ParietalCortexService(BaseCosmosService):
    """
    P-5: Parietal Cortex
    Analytical processing, spatial reasoning, mathematical processing.
    
    Brodmann Areas: 5, 7, 39, 40
    Neurotransmitters: Glutamate
    """
    
    def __init__(self, config: ServiceConfig, n_neurons: int = 120):
        super().__init__(config)
        self.population = NeuralPopulation(
            "Parietal Cortex",
            n_neurons,
            NeuronType.EXCITATORY
        )
        
    async def initialize(self) -> None:
        self.log('info', 'Parietal Cortex Service initialized')
        self.initialized = True
        
    async def process(self, message: ServiceMessage) -> Optional[ServiceMessage]:
        if message.type == 'EXECUTIVE_COORDINATION':
            coordination = message.payload
            
            # Analytical processing
            input_currents = np.random.randn(self.population.n_neurons) * 4.0
            spikes = self.population.step(input_currents)
            
            firing_rate = self.population.get_firing_rate()
            
            analysis = {
                'analytical_depth': firing_rate * 10,
                'spatial_reasoning': firing_rate * 8,
                'processed_coordination': coordination,
                'ready_for_output': firing_rate > 0.3
            }
            
            return create_message(
                'ANALYTICAL_RESULT',
                analysis,
                self.config.service_name,
                'cerebral:O-4'
            )
        return None
    
    async def shutdown(self) -> None:
        self.log('info', 'Parietal Cortex Service shutdown')


class MotorCortexService(BaseCosmosService):
    """
    O-4: Motor Cortex
    Action planning, motor sequencing, formatted output.
    
    Brodmann Areas: 4, 6
    Neurotransmitters: Glutamate, Acetylcholine
    """
    
    def __init__(self, config: ServiceConfig, n_neurons: int = 100):
        super().__init__(config)
        self.population = NeuralPopulation(
            "Motor Cortex",
            n_neurons,
            NeuronType.EXCITATORY
        )
        
    async def initialize(self) -> None:
        self.log('info', 'Motor Cortex Service initialized')
        self.initialized = True
        
    async def process(self, message: ServiceMessage) -> Optional[ServiceMessage]:
        if message.type == 'ANALYTICAL_RESULT':
            analysis = message.payload
            
            # Generate motor plan
            input_currents = np.random.randn(self.population.n_neurons) * 4.5
            spikes = self.population.step(input_currents)
            
            firing_rate = self.population.get_firing_rate()
            
            motor_plan = {
                'action_strength': firing_rate * 10,
                'motor_sequence': list(spikes[:10]),  # First 10 neurons
                'execution_ready': firing_rate > 0.4,
                'analysis_basis': analysis
            }
            
            return create_message(
                'MOTOR_OUTPUT',
                motor_plan,
                self.config.service_name
            )
        return None
    
    async def shutdown(self) -> None:
        self.log('info', 'Motor Cortex Service shutdown')


# ============================================================================
# SOMATIC TRIAD: BASAL GANGLIA MOTOR CONTROL
# ============================================================================

class StriatumService(BaseCosmosService):
    """
    M-1: Striatum (Caudate, Putamen)
    Action selection, habit formation.
    
    Neurotransmitters: GABA, Dopamine receptors
    """
    
    def __init__(self, config: ServiceConfig, n_neurons: int = 150):
        super().__init__(config)
        self.population = NeuralPopulation(
            "Striatum",
            n_neurons,
            NeuronType.INHIBITORY
        )
        self.action_values = {}
        
    async def initialize(self) -> None:
        self.log('info', 'Striatum Service initialized')
        self.initialized = True
        
    async def process(self, message: ServiceMessage) -> Optional[ServiceMessage]:
        if message.type == 'MOTOR_OUTPUT':
            motor_plan = message.payload
            
            # Action selection
            input_currents = np.random.randn(self.population.n_neurons) * 3.5
            spikes = self.population.step(input_currents)
            
            firing_rate = self.population.get_firing_rate()
            
            action_selection = {
                'selected_action': 'execute' if firing_rate > 0.3 else 'inhibit',
                'action_value': firing_rate * 10,
                'habit_strength': firing_rate * 5,
                'motor_plan': motor_plan
            }
            
            return create_message(
                'ACTION_SELECTED',
                action_selection,
                self.config.service_name
            )
        return None
    
    async def shutdown(self) -> None:
        self.log('info', 'Striatum Service shutdown')


class ThalamusService(BaseCosmosService):
    """
    S-8: Thalamus (VL, VA, MD nuclei)
    Sensory relay, motor gating.
    
    Neurotransmitters: Glutamate
    """
    
    def __init__(self, config: ServiceConfig, n_neurons: int = 100):
        super().__init__(config)
        self.population = NeuralPopulation(
            "Thalamus",
            n_neurons,
            NeuronType.EXCITATORY
        )
        
    async def initialize(self) -> None:
        self.log('info', 'Thalamus Service initialized')
        self.initialized = True
        
    async def process(self, message: ServiceMessage) -> Optional[ServiceMessage]:
        if message.type == 'SENSORY_INPUT':
            # Relay sensory information
            sensory_data = message.payload
            
            input_currents = np.random.randn(self.population.n_neurons) * 4.0
            spikes = self.population.step(input_currents)
            
            firing_rate = self.population.get_firing_rate()
            
            relayed_info = {
                'relay_strength': firing_rate * 10,
                'gating_active': firing_rate > 0.4,
                'sensory_data': sensory_data
            }
            
            return create_message(
                'RELAYED_SENSORY',
                relayed_info,
                self.config.service_name,
                'cerebral:T-7'
            )
        return None
    
    async def shutdown(self) -> None:
        self.log('info', 'Thalamus Service shutdown')


# ============================================================================
# AUTONOMIC TRIAD: LIMBIC SYSTEM REGULATION
# ============================================================================

class HypothalamusService(BaseCosmosService):
    """
    M-1: Hypothalamus (PVN, LH, VMH)
    Homeostatic monitoring, autonomic control.
    
    Neurotransmitters: Multiple peptides
    """
    
    def __init__(self, config: ServiceConfig, n_neurons: int = 60):
        super().__init__(config)
        self.population = NeuralPopulation(
            "Hypothalamus",
            n_neurons,
            NeuronType.MODULATORY
        )
        self.homeostatic_setpoints = {
            'arousal': 0.5,
            'stress': 0.3,
            'energy': 0.7
        }
        
    async def initialize(self) -> None:
        self.log('info', 'Hypothalamus Service initialized')
        self.initialized = True
        
    async def process(self, message: ServiceMessage) -> Optional[ServiceMessage]:
        # Monitor system state
        input_currents = np.random.randn(self.population.n_neurons) * 2.0
        spikes = self.population.step(input_currents)
        
        firing_rate = self.population.get_firing_rate()
        
        homeostatic_state = {
            'arousal_level': firing_rate,
            'stress_level': max(0, firing_rate - 0.5),
            'energy_level': 1.0 - firing_rate,
            'autonomic_balance': 'sympathetic' if firing_rate > 0.5 else 'parasympathetic'
        }
        
        return create_message(
            'HOMEOSTATIC_STATE',
            homeostatic_state,
            self.config.service_name
        )
    
    async def shutdown(self) -> None:
        self.log('info', 'Hypothalamus Service shutdown')


class HippocampusService(BaseCosmosService):
    """
    S-8: Hippocampus (CA1, CA3, DG)
    Episodic memory, spatial context.
    
    Neurotransmitters: Glutamate, Acetylcholine
    """
    
    def __init__(self, config: ServiceConfig, n_neurons: int = 200):
        super().__init__(config)
        self.population = NeuralPopulation(
            "Hippocampus",
            n_neurons,
            NeuronType.EXCITATORY
        )
        self.memory_buffer = []
        self.max_memories = 100
        
    async def initialize(self) -> None:
        self.log('info', 'Hippocampus Service initialized')
        self.initialized = True
        
    async def process(self, message: ServiceMessage) -> Optional[ServiceMessage]:
        # Store and retrieve memories
        input_currents = np.random.randn(self.population.n_neurons) * 3.0
        spikes = self.population.step(input_currents)
        
        firing_rate = self.population.get_firing_rate()
        
        # Store message in memory
        self.memory_buffer.append({
            'timestamp': message.timestamp,
            'type': message.type,
            'firing_rate': firing_rate
        })
        
        if len(self.memory_buffer) > self.max_memories:
            self.memory_buffer.pop(0)
        
        memory_state = {
            'memory_strength': firing_rate * 10,
            'context_richness': len(self.memory_buffer),
            'recent_memories': self.memory_buffer[-5:]
        }
        
        return create_message(
            'MEMORY_STATE',
            memory_state,
            self.config.service_name
        )
    
    async def shutdown(self) -> None:
        self.log('info', 'Hippocampus Service shutdown')


# ============================================================================
# COGNITIVE SYSTEM
# ============================================================================

class CognitiveNeuralSystem:
    """
    Complete Cosmos System 5 Cognitive Neural Network.
    
    Integrates all three triads:
    - Cerebral: Neocortex executive functions
    - Somatic: Basal ganglia motor control
    - Autonomic: Limbic system regulation
    """
    
    def __init__(self):
        self.coordinator = TriadicCoordinator()
        self.services = {}
        
    async def initialize(self):
        """Initialize all brain region services."""
        # Cerebral Triad
        pfc = PrefrontalCortexService(
            ServiceConfig(
                "prefrontal-cortex",
                Triad.CEREBRAL,
                ServicePosition.T7,
                Polarity.SYMPATHETIC,
                Dimension.POTENTIAL
            )
        )
        await pfc.initialize()
        self.coordinator.register_service(pfc)
        self.services['pfc'] = pfc
        
        acc = AnteriorCingulateService(
            ServiceConfig(
                "anterior-cingulate",
                Triad.CEREBRAL,
                ServicePosition.PD2,
                Polarity.PARASYMPATHETIC,
                Dimension.POTENTIAL
            )
        )
        await acc.initialize()
        self.coordinator.register_service(acc)
        self.services['acc'] = acc
        
        parietal = ParietalCortexService(
            ServiceConfig(
                "parietal-cortex",
                Triad.CEREBRAL,
                ServicePosition.P5,
                Polarity.SOMATIC,
                Dimension.COMMITMENT
            )
        )
        await parietal.initialize()
        self.coordinator.register_service(parietal)
        self.services['parietal'] = parietal
        
        motor = MotorCortexService(
            ServiceConfig(
                "motor-cortex",
                Triad.CEREBRAL,
                ServicePosition.O4,
                Polarity.SOMATIC,
                Dimension.COMMITMENT
            )
        )
        await motor.initialize()
        self.coordinator.register_service(motor)
        self.services['motor'] = motor
        
        # Somatic Triad
        striatum = StriatumService(
            ServiceConfig(
                "striatum",
                Triad.SOMATIC,
                ServicePosition.M1,
                Polarity.SYMPATHETIC,
                Dimension.PERFORMANCE
            )
        )
        await striatum.initialize()
        self.coordinator.register_service(striatum)
        self.services['striatum'] = striatum
        
        thalamus = ThalamusService(
            ServiceConfig(
                "thalamus",
                Triad.SOMATIC,
                ServicePosition.S8,
                Polarity.SOMATIC,
                Dimension.PERFORMANCE
            )
        )
        await thalamus.initialize()
        self.coordinator.register_service(thalamus)
        self.services['thalamus'] = thalamus
        
        # Autonomic Triad
        hypothalamus = HypothalamusService(
            ServiceConfig(
                "hypothalamus",
                Triad.AUTONOMIC,
                ServicePosition.M1,
                Polarity.PARASYMPATHETIC,
                Dimension.PERFORMANCE
            )
        )
        await hypothalamus.initialize()
        self.coordinator.register_service(hypothalamus)
        self.services['hypothalamus'] = hypothalamus
        
        hippocampus = HippocampusService(
            ServiceConfig(
                "hippocampus",
                Triad.AUTONOMIC,
                ServicePosition.S8,
                Polarity.PARASYMPATHETIC,
                Dimension.PERFORMANCE
            )
        )
        await hippocampus.initialize()
        self.coordinator.register_service(hippocampus)
        self.services['hippocampus'] = hippocampus
        
        print("✓ Cognitive Neural System initialized")
        print(f"  - Cerebral Triad: {len([s for s in self.services.values() if s.config.triad == Triad.CEREBRAL])} regions")
        print(f"  - Somatic Triad: {len([s for s in self.services.values() if s.config.triad == Triad.SOMATIC])} regions")
        print(f"  - Autonomic Triad: {len([s for s in self.services.values() if s.config.triad == Triad.AUTONOMIC])} regions")
    
    async def process_cognitive_task(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """Process a cognitive task through the neural system."""
        # Create sensory input message
        sensory_msg = create_message('SENSORY_INPUT', task_input, 'external')
        
        # Process through thalamus (sensory relay)
        relayed_msg = await self.services['thalamus'].process(sensory_msg)
        if not relayed_msg:
            return {'error': 'Thalamus processing failed'}
        
        # Process through prefrontal cortex (creative ideation)
        ideas_msg = await self.services['pfc'].process(relayed_msg)
        if not ideas_msg:
            return {'error': 'PFC processing failed'}
        
        # Process through anterior cingulate (executive coordination)
        coord_msg = await self.services['acc'].process(ideas_msg)
        if not coord_msg:
            return {'error': 'ACC processing failed'}
        
        # Process through parietal cortex (analytical processing)
        analysis_msg = await self.services['parietal'].process(coord_msg)
        if not analysis_msg:
            return {'error': 'Parietal processing failed'}
        
        # Process through motor cortex (action planning)
        motor_msg = await self.services['motor'].process(analysis_msg)
        if not motor_msg:
            return {'error': 'Motor cortex processing failed'}
        
        # Process through striatum (action selection)
        action_msg = await self.services['striatum'].process(motor_msg)
        
        # Monitor homeostatic state
        homeo_msg = await self.services['hypothalamus'].process(sensory_msg)
        
        # Update memory
        if action_msg:
            memory_msg = await self.services['hippocampus'].process(action_msg)
        else:
            memory_msg = None
        
        return {
            'action': action_msg.payload if action_msg else None,
            'homeostasis': homeo_msg.payload if homeo_msg else None,
            'memory': memory_msg.payload if memory_msg else None
        }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    import asyncio
    
    setup_logging("INFO")
    
    print("=== COSYS-XNN: Cognitive Neural Network Demo ===\n")
    
    # Create system
    system = CognitiveNeuralSystem()
    asyncio.run(system.initialize())
    
    # Process cognitive task
    print("\n--- Processing cognitive task ---")
    task = {
        'context': 'problem-solving',
        'complexity': 'high',
        'sensory_input': np.random.randn(10)
    }
    
    result = asyncio.run(system.process_cognitive_task(task))
    
    print("\n=== Results ===")
    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        if result.get('action'):
            print(f"Action: {result['action']['selected_action']}")
            print(f"Action Value: {result['action']['action_value']:.2f}")
        if result.get('homeostasis'):
            print(f"Arousal: {result['homeostasis']['arousal_level']:.2f}")
        if result.get('memory'):
            print(f"Memory Strength: {result['memory']['memory_strength']:.2f}")
    
    print("\n✓ COSYS-XNN demonstration complete")
