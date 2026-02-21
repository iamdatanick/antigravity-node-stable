"""PHUC Stack Specialist Agents.

Provides specialized agents for each component of the PHUC
(Pharmaceutical HCP/Unbranded Content) marketing intelligence platform.
"""

from .airflow_agent import AirflowAgent, AirflowConfig
from .base import SpecialistAgent, SpecialistCapability, SpecialistConfig, SpecialistResult
from .cloudflare_agent import CloudflareAgent, CloudflareConfig
from .kafka_agent import KafkaAgent, KafkaConfig
from .keycloak_agent import KeycloakAgent, KeycloakConfig
from .langchain_agent import LangChainAgent, LangChainConfig
from .mcp_agent import MCPSpecialistAgent, MCPSpecialistConfig
from .milvus_agent import MilvusAgent, MilvusConfig
from .seaweed_agent import SeaweedAgent, SeaweedConfig
from .openmetadata_agent import OpenMetadataAgent, OpenMetadataConfig
from .ovms_agent import OVMSAgent, OVMSConfig
from .phuc_platform_agent import PHUCConfig, PHUCOperation, PHUCPlatformAgent
from .starrocks_agent import StarRocksAgent, StarRocksConfig
from .supervisor import (
    RoutingStrategy,
    StackSupervisor,
    SupervisorConfig,
    create_phuc_stack_supervisor,
)
from .unomi_agent import UnomiAgent, UnomiConfig
from .vault_agent import VaultAgent, VaultConfig

__all__ = [
    # Base
    "SpecialistAgent",
    "SpecialistConfig",
    "SpecialistCapability",
    "SpecialistResult",
    # Supervisor
    "StackSupervisor",
    "SupervisorConfig",
    "RoutingStrategy",
    "create_phuc_stack_supervisor",
    # Storage
    "SeaweedAgent",
    "SeaweedConfig",
    # Messaging
    "KafkaAgent",
    "KafkaConfig",
    # Analytics
    "StarRocksAgent",
    "StarRocksConfig",
    # CDP
    "UnomiAgent",
    "UnomiConfig",
    # Vector DB
    "MilvusAgent",
    "MilvusConfig",
    # ML Serving
    "OVMSAgent",
    "OVMSConfig",
    # Orchestration
    "AirflowAgent",
    "AirflowConfig",
    # Governance
    "OpenMetadataAgent",
    "OpenMetadataConfig",
    # AI/LLM
    "LangChainAgent",
    "LangChainConfig",
    "MCPSpecialistAgent",
    "MCPSpecialistConfig",
    # Security
    "KeycloakAgent",
    "KeycloakConfig",
    "VaultAgent",
    "VaultConfig",
    "CloudflareAgent",
    "CloudflareConfig",
    # PHUC Platform
    "PHUCPlatformAgent",
    "PHUCConfig",
    "PHUCOperation",
]
