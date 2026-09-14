"""
Sword Coding Gym: Docker Execution & Benchmarking for Coding RL.
Supports GAIR/OpenSWE and AweAI-Team/Scale-SWE schemas and container harnesses.
"""

from sword.gym.schema import GymInstance, GymExecutionResult
from sword.gym.adapters import OpenSWEAdapter, ScaleSWEAdapter, load_swe_dataset
from sword.gym.docker_gym import DockerCodingGym, extract_unified_diff
from sword.gym.server import GymServer, ZrokTunnelManager
from sword.gym.client import RemoteCodingGym

__all__ = [
    "GymInstance",
    "GymExecutionResult",
    "OpenSWEAdapter",
    "ScaleSWEAdapter",
    "load_swe_dataset",
    "DockerCodingGym",
    "extract_unified_diff",
    "GymServer",
    "ZrokTunnelManager",
    "RemoteCodingGym",
]
