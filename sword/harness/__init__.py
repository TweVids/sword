"""
Harness package: CLI agentic tool hooks for OpenCode, Codex, Gemini, and Pi.
"""

from sword.harness.base import BaseHarness, HarnessResult
from sword.harness.sandbox import WorkspaceSandbox
from sword.harness.prompts import get_harness_system_prompt, ensure_harness_repo
from sword.harness.opencode import OpenCodeHarness
from sword.harness.codex_cli import CodexCliHarness
from sword.harness.gemini_cli import GeminiCliHarness
from sword.harness.pi_agent import PiHarness
from sword.harness.registry import get_harness, HARNESS_REGISTRY

__all__ = [
    "BaseHarness",
    "HarnessResult",
    "WorkspaceSandbox",
    "get_harness_system_prompt",
    "ensure_harness_repo",
    "OpenCodeHarness",
    "CodexCliHarness",
    "GeminiCliHarness",
    "PiHarness",
    "get_harness",
    "HARNESS_REGISTRY",
]
