"""
COSYS-XNN: Cognitive Neural Network — Unified Orchestrator
==========================================================

Cosmos System 5 model applied to cognitive function, brain regions, and neural networks.
Implements the complete 18-service [[D-T]-[P-O]-[S-M]] pattern across three triads.

Author: Cosmos System Enhancement Project
License: AGPL-3.0
"""

import os
import sys
import asyncio
import numpy as np
from typing import Optional, Dict, Any, List

# ── cosmos_core: use local stub, fall back to shared-lib env var ──────────────
sys.path.insert(0, os.path.dirname(__file__))
from cosmos_core import (
    BaseCosmosService, ServiceConfig, ServiceMessage,
    Triad, Polarity, ServicePosition, Dimension,
    TriadicCoordinator, create_message, setup_logging,
)

# ── Cerebral Triad ────────────────────────────────────────────────────────────
from cerebral_triad.prefrontal_cortex  import PrefrontalCortexService
from cerebral_triad.anterior_cingulate import AnteriorCingulateService
from cerebral_triad.parietal_cortex    import ParietalCortexService
from cerebral_triad.motor_cortex       import MotorCortexService

# ── Somatic Triad ─────────────────────────────────────────────────────────────
from somatic_triad.striatum        import StriatumService
from somatic_triad.thalamus        import ThalamusService
from somatic_triad.caudate_nucleus import CaudateNucleusService
from somatic_triad.putamen         import PutamenService
from somatic_triad.globus_pallidus import GlobusPallidusService
from somatic_triad.substantia_nigra import SubstantiaNigraService

# ── Autonomic Triad ───────────────────────────────────────────────────────────
from autonomic_triad.hypothalamus      import HypothalamusService
from autonomic_triad.hippocampus       import HippocampusService
from autonomic_triad.amygdala          import AmygdalaService
from autonomic_triad.brainstem         import BrainstemService
from autonomic_triad.insula            import InsulaService
from autonomic_triad.cingulate_autonomic import CingulateAutonomicService

# ── Integration Hub ───────────────────────────────────────────────────────────
from integration_hub.global_workspace import GlobalWorkspace, Representation
from integration_hub.event_bus        import EventBus
from integration_hub.triple_network   import TripleNetworkModel

# ── Cognitive Core ────────────────────────────────────────────────────────────
from cognitive_core.autognosis import AutognosisOrchestrator

# ── Models ────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
from system5_neural import NeuralSystem5StateMachine


# =============================================================================
# COGNITIVE NEURAL SYSTEM
# =============================================================================

class CognitiveNeuralSystem:
    """
    Complete Cosmos System 5 Cognitive Neural Network.

    Integrates all 18 services across three triads:
      Cerebral  [4]: PFC (T-7), ACC (PD-2), Parietal (P-5), Motor (O-4)
      Somatic   [6]: Striatum (M-1), Thalamus (S-8), Caudate (PD-2),
                     Putamen (T-7), GlobusPallidus (P-5), SubstantiaNigra (O-4)
      Autonomic [6]: Hypothalamus (M-1), Hippocampus (S-8), Amygdala (PD-2),
                     Brainstem (T-7), Insula (P-5), CingulateAutonomic (O-4)
    """

    def __init__(self):
        self.coordinator   = TriadicCoordinator()
        self.services: Dict[str, BaseCosmosService] = {}
        self.state_machine = NeuralSystem5StateMachine()
        self.global_workspace = GlobalWorkspace(ignition_threshold=0.7)
        self.event_bus     = EventBus()
        self.triple_network = TripleNetworkModel()
        self.autognosis    = AutognosisOrchestrator()

    # ── Initialisation ────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Initialise and register all 18 brain-region services."""
        await self._init_cerebral_triad()
        await self._init_somatic_triad()
        await self._init_autonomic_triad()

        for name in self.services:
            self.global_workspace.register_service(name)

        print("✓ CognitiveNeuralSystem initialised")
        print(f"  Cerebral  triad: {len(self.get_triad_services(Triad.CEREBRAL))} services")
        print(f"  Somatic   triad: {len(self.get_triad_services(Triad.SOMATIC))} services")
        print(f"  Autonomic triad: {len(self.get_triad_services(Triad.AUTONOMIC))} services")
        print(f"  Total services : {len(self.services)}")

    async def _init_cerebral_triad(self) -> None:
        specs = [
            ("pfc",      PrefrontalCortexService,  ServicePosition.T7,  Polarity.SYMPATHETIC,     Dimension.POTENTIAL),
            ("acc",      AnteriorCingulateService,  ServicePosition.PD2, Polarity.PARASYMPATHETIC, Dimension.POTENTIAL),
            ("parietal", ParietalCortexService,     ServicePosition.P5,  Polarity.SOMATIC,         Dimension.COMMITMENT),
            ("motor",    MotorCortexService,         ServicePosition.O4,  Polarity.SOMATIC,         Dimension.COMMITMENT),
        ]
        for key, cls, pos, pol, dim in specs:
            svc = cls(ServiceConfig(key, Triad.CEREBRAL, pos, pol, dim))
            await svc.initialize()
            self.coordinator.register_service(svc)
            self.services[key] = svc

    async def _init_somatic_triad(self) -> None:
        specs = [
            ("striatum",         StriatumService,        ServicePosition.M1,  Polarity.SYMPATHETIC,     Dimension.PERFORMANCE),
            ("thalamus",         ThalamusService,         ServicePosition.S8,  Polarity.SOMATIC,         Dimension.PERFORMANCE),
            ("caudate",          CaudateNucleusService,   ServicePosition.PD2, Polarity.PARASYMPATHETIC, Dimension.PERFORMANCE),
            ("putamen",          PutamenService,          ServicePosition.T7,  Polarity.SYMPATHETIC,     Dimension.PERFORMANCE),
            ("globus_pallidus",  GlobusPallidusService,   ServicePosition.P5,  Polarity.SOMATIC,         Dimension.PERFORMANCE),
            ("substantia_nigra", SubstantiaNigraService,  ServicePosition.O4,  Polarity.SYMPATHETIC,     Dimension.PERFORMANCE),
        ]
        for key, cls, pos, pol, dim in specs:
            svc = cls(ServiceConfig(key, Triad.SOMATIC, pos, pol, dim))
            await svc.initialize()
            self.coordinator.register_service(svc)
            self.services[key] = svc

    async def _init_autonomic_triad(self) -> None:
        specs = [
            ("hypothalamus",       HypothalamusService,       ServicePosition.M1,  Polarity.PARASYMPATHETIC, Dimension.PERFORMANCE),
            ("hippocampus",        HippocampusService,         ServicePosition.S8,  Polarity.PARASYMPATHETIC, Dimension.PERFORMANCE),
            ("amygdala",           AmygdalaService,            ServicePosition.PD2, Polarity.SYMPATHETIC,     Dimension.PERFORMANCE),
            ("brainstem",          BrainstemService,           ServicePosition.T7,  Polarity.SYMPATHETIC,     Dimension.PERFORMANCE),
            ("insula",             InsulaService,              ServicePosition.P5,  Polarity.SOMATIC,         Dimension.PERFORMANCE),
            ("cingulate_autonomic", CingulateAutonomicService, ServicePosition.O4,  Polarity.PARASYMPATHETIC, Dimension.PERFORMANCE),
        ]
        for key, cls, pos, pol, dim in specs:
            svc = cls(ServiceConfig(key, Triad.AUTONOMIC, pos, pol, dim))
            await svc.initialize()
            self.coordinator.register_service(svc)
            self.services[key] = svc

    # ── Cognitive Pipeline ────────────────────────────────────────────────────

    async def process_cognitive_task(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a cognitive task through the full neural pipeline.
        Pipeline: thalamus → pfc → acc → parietal → motor → striatum
        Parallel: hypothalamus, hippocampus, amygdala monitor throughout.
        """
        sm_state = self.state_machine.step()
        sensory_msg = create_message("SENSORY_INPUT", task_input, "external")

        # ── Main pipeline ──────────────────────────────────────────────────
        relayed_msg  = await self.services["thalamus"].process(sensory_msg)
        if not relayed_msg:
            return {"error": "Thalamus processing failed"}

        ideas_msg    = await self.services["pfc"].process(relayed_msg)
        if not ideas_msg:
            return {"error": "PFC processing failed"}

        coord_msg    = await self.services["acc"].process(ideas_msg)
        if not coord_msg:
            return {"error": "ACC processing failed"}

        analysis_msg = await self.services["parietal"].process(coord_msg)
        if not analysis_msg:
            return {"error": "Parietal processing failed"}

        motor_msg    = await self.services["motor"].process(analysis_msg)
        if not motor_msg:
            return {"error": "Motor cortex processing failed"}

        action_msg   = await self.services["striatum"].process(motor_msg)

        # ── Extended somatic path ──────────────────────────────────────────
        if action_msg:
            caudate_msg = await self.services["caudate"].process(action_msg)
            if caudate_msg:
                putamen_msg = await self.services["putamen"].process(caudate_msg)
                if putamen_msg:
                    gp_msg = await self.services["globus_pallidus"].process(putamen_msg)
                    if gp_msg:
                        await self.services["substantia_nigra"].process(gp_msg)

        # ── Autonomic parallel processes ───────────────────────────────────
        homeo_msg   = await self.services["hypothalamus"].process(sensory_msg)
        memory_msg  = await self.services["hippocampus"].process(action_msg or sensory_msg)
        emo_msg     = await self.services["amygdala"].process(sensory_msg)
        arousal_msg = None
        if emo_msg:
            arousal_msg = await self.services["brainstem"].process(emo_msg)
        intero_msg  = None
        if emo_msg:
            intero_msg = await self.services["insula"].process(emo_msg)
        viscero_msg = None
        if intero_msg:
            viscero_msg = await self.services["cingulate_autonomic"].process(intero_msg)

        # ── Global Workspace competition ───────────────────────────────────
        candidates = []
        for key, svc in self.services.items():
            state = self.get_triad_state(svc.config.triad)
            rate = state.get(key, {}).get("firing_rate", 0.0)
            candidates.append(Representation(source=key, content=key, salience=rate))
        broadcast = self.global_workspace.compete_for_access(candidates)

        # ── Triple network update ──────────────────────────────────────────
        avg_salience = float(np.mean([c.salience for c in candidates])) if candidates else 0.3
        network_state = self.triple_network.switch_networks(avg_salience)

        return {
            "action":        action_msg.payload   if action_msg   else None,
            "homeostasis":   homeo_msg.payload     if homeo_msg    else None,
            "memory":        memory_msg.payload    if memory_msg   else None,
            "emotion":       emo_msg.payload       if emo_msg      else None,
            "arousal":       arousal_msg.payload   if arousal_msg  else None,
            "interoception": intero_msg.payload    if intero_msg   else None,
            "visceromotor":  viscero_msg.payload   if viscero_msg  else None,
            "state_machine": sm_state,
            "network_mode":  network_state["mode"],
            "broadcast":     broadcast.representation.source if broadcast else None,
        }

    # ── Triad State Accessors ─────────────────────────────────────────────────

    def get_triad_services(self, triad: Triad) -> Dict[str, BaseCosmosService]:
        return {k: v for k, v in self.services.items() if v.config.triad == triad}

    def get_triad_state(self, triad: Triad) -> Dict[str, Any]:
        """Return firing rates and activity for all services in a triad."""
        state = {}
        for key, svc in self.get_triad_services(triad).items():
            if hasattr(svc, "population"):
                state[key] = {
                    "firing_rate": svc.population.get_firing_rate(),
                    "n_neurons":   svc.population.n_neurons,
                    "neuron_type": svc.population.neuron_type.value,
                }
            elif hasattr(svc, "bla"):  # Amygdala
                state[key] = {
                    "bla_rate": svc.bla.get_firing_rate(),
                    "cea_rate": svc.cea.get_firing_rate(),
                    "firing_rate": (svc.bla.get_firing_rate() + svc.cea.get_firing_rate()) / 2,
                }
            elif hasattr(svc, "pag"):  # Brainstem
                state[key] = {
                    "pag_rate": svc.pag.get_firing_rate(),
                    "lc_rate":  svc.lc.get_firing_rate(),
                    "nts_rate": svc.nts.get_firing_rate(),
                    "firing_rate": svc.lc.get_firing_rate(),
                }
            elif hasattr(svc, "anterior"):  # Insula
                state[key] = {
                    "anterior_rate": svc.anterior.get_firing_rate(),
                    "posterior_rate": svc.posterior.get_firing_rate(),
                    "firing_rate": (svc.anterior.get_firing_rate() + svc.posterior.get_firing_rate()) / 2,
                }
        return state

    def get_cerebral_state(self)  -> Dict[str, Any]: return self.get_triad_state(Triad.CEREBRAL)
    def get_somatic_state(self)   -> Dict[str, Any]: return self.get_triad_state(Triad.SOMATIC)
    def get_autonomic_state(self) -> Dict[str, Any]: return self.get_triad_state(Triad.AUTONOMIC)

    def get_service_firing_rates(self) -> Dict[str, float]:
        rates = {}
        for key, svc in self.services.items():
            triad_state = self.get_triad_state(svc.config.triad)
            if key in triad_state:
                rates[key] = triad_state[key].get("firing_rate", 0.0)
        return rates

    async def run_autognosis_cycle(self) -> Dict[str, Any]:
        """Run a full self-awareness cycle across the system."""
        rates = self.get_service_firing_rates()
        return await self.autognosis.run_autognosis_cycle(rates)

    async def shutdown(self) -> None:
        for svc in self.services.values():
            await svc.shutdown()


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    setup_logging("INFO")
    print("=== COSYS-XNN: Cognitive Neural Network Demo ===\n")

    system = CognitiveNeuralSystem()
    asyncio.run(system.initialize())

    print("\n--- Processing cognitive task ---")
    task = {"context": "problem-solving", "complexity": "high"}
    result = asyncio.run(system.process_cognitive_task(task))

    print("\n=== Results ===")
    if result.get("action"):
        print(f"Action:       {result['action'].get('selected_action')}")
    if result.get("homeostasis"):
        print(f"Arousal:      {result['homeostasis'].get('arousal_level', 0):.3f}")
    if result.get("memory"):
        print(f"Memory:       {result['memory'].get('memory_strength', 0):.3f}")
    if result.get("emotion"):
        print(f"Valence:      {result['emotion'].get('emotional_valence', 0):.3f}")
    print(f"Network mode: {result.get('network_mode')}")
    print(f"SM step:      {result.get('state_machine', {}).get('t', 0)}")

    print("\n=== Triad States ===")
    print("Cerebral: ", {k: f"{v.get('firing_rate', 0):.3f}" for k, v in system.get_cerebral_state().items()})
    print("Somatic:  ", {k: f"{v.get('firing_rate', 0):.3f}" for k, v in system.get_somatic_state().items()})
    print("Autonomic:", {
        k: f"{v.get('firing_rate', 0):.3f}" if "firing_rate" in v else str(v)
        for k, v in system.get_autonomic_state().items()
    })

    print("\n--- Running autognosis cycle ---")
    autognosis_result = asyncio.run(system.run_autognosis_cycle())
    print(f"Insights: {len(autognosis_result['insights'])}")
    for i in autognosis_result["insights"]:
        print(f"  - {i['insight']}")

    print("\n✓ COSYS-XNN demonstration complete")
