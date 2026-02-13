"""PHUC Stack Specialist Agents.

Provides specialized agents for each component of the PHUC
(Pharmaceutical HCP/Unbranded Content) marketing intelligence platform.
"""

from .base import SpecialistAgent, SpecialistConfig, SpecialistCapability, SpecialistResult
from .supervisor import StackSupervisor, SupervisorConfig, RoutingStrategy, create_phuc_stack_supervisor
from .minio_agent import MinIOAgent, MinIOConfig
from .kafka_agent import KafkaAgent, KafkaConfig
from .starrocks_agent import StarRocksAgent, StarRocksConfig
from .unomi_agent import UnomiAgent, UnomiConfig
from .milvus_agent import MilvusAgent, MilvusConfig
from .ovms_agent import OVMSAgent, OVMSConfig
from .airflow_agent import AirflowAgent, AirflowConfig
from .openmetadata_agent import OpenMetadataAgent, OpenMetadataConfig
from .langchain_agent import LangChainAgent, LangChainConfig
from .mcp_agent import MCPSpecialistAgent, MCPSpecialistConfig
from .keycloak_agent import KeycloakAgent, KeycloakConfig
from .vault_agent import VaultAgent, VaultConfig
from .cloudflare_agent import CloudflareAgent, CloudflareConfig
from .phuc_platform_agent import PHUCPlatformAgent, PHUCConfig, PHUCOperation
from .phuc_pharma_agent import (
    PharmaDataAgent, PharmaConfig, NPIRecord, NDCRecord, 
    DoctorProfile, IdentityType, AttributionModel
)

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
    "MinIOAgent",
    "MinIOConfig",
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
    # PHUC Pharma Data
    "PharmaDataAgent",
    "PharmaConfig",
    "NPIRecord",
    "NDCRecord",
    "DoctorProfile",
    "IdentityType",
    "AttributionModel",
]
