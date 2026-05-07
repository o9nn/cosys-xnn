"""
cosmos_core stub — local implementation of the cosmos_core shared library interface.
The original library lives at https://github.com/o9nn/cosmos-system-5.
This stub replicates its public API so cosys-xnn works standalone.
"""

import logging
import uuid
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Triad(Enum):
    CEREBRAL = "cerebral"
    SOMATIC = "somatic"
    AUTONOMIC = "autonomic"


class Polarity(Enum):
    SYMPATHETIC = "sympathetic"
    PARASYMPATHETIC = "parasympathetic"
    SOMATIC = "somatic"


class ServicePosition(Enum):
    T7 = "T-7"
    PD2 = "PD-2"
    P5 = "P-5"
    O4 = "O-4"
    S8 = "S-8"
    M1 = "M-1"


class Dimension(Enum):
    POTENTIAL = "potential"
    COMMITMENT = "commitment"
    PERFORMANCE = "performance"


@dataclass
class ServiceConfig:
    service_name: str
    triad: Triad
    position: ServicePosition
    polarity: Polarity
    dimension: Dimension


@dataclass
class ServiceMessage:
    type: str
    payload: Any
    source: str
    target: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))


def create_message(
    msg_type: str,
    payload: Any,
    source: str,
    target: Optional[str] = None,
) -> ServiceMessage:
    return ServiceMessage(type=msg_type, payload=payload, source=source, target=target)


class BaseCosmosService:
    def __init__(self, config: ServiceConfig):
        self.config = config
        self.initialized = False
        self._logger = logging.getLogger(config.service_name)

    def log(self, level: str, msg: str) -> None:
        getattr(self._logger, level)(msg)

    async def initialize(self) -> None:
        raise NotImplementedError

    async def process(self, message: ServiceMessage) -> Optional[ServiceMessage]:
        raise NotImplementedError

    async def shutdown(self) -> None:
        raise NotImplementedError


class TriadicCoordinator:
    def __init__(self):
        self._services: List[BaseCosmosService] = []

    def register_service(self, service: BaseCosmosService) -> None:
        self._services.append(service)

    def get_services(self) -> List[BaseCosmosService]:
        return list(self._services)

    def get_services_by_triad(self, triad: Triad) -> List[BaseCosmosService]:
        return [s for s in self._services if s.config.triad == triad]


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
