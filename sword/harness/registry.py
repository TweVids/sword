"""
Harness Registry: Resolves agentic CLI harness by dataset attribute name.
"""

from typing import Optional, Dict, Type

from sword.harness.base import BaseHarness
from sword.harness.opencode import OpenCodeHarness
from sword.harness.codex_cli import CodexCliHarness
from sword.harness.gemini_cli import GeminiCliHarness
from sword.harness.pi_agent import PiHarness


HARNESS_REGISTRY: Dict[str, Type[BaseHarness]] = {
    "opencode": OpenCodeHarness,
    "open-code": OpenCodeHarness,
    "opencodeinterpreter": OpenCodeHarness,
    "opendevin": OpenCodeHarness,
    "codex": CodexCliHarness,
    "codex_cli": CodexCliHarness,
    "codex-cli": CodexCliHarness,
    "gemini": GeminiCliHarness,
    "gemini_cli": GeminiCliHarness,
    "gemini-cli": GeminiCliHarness,
    "pi": PiHarness,
    "pi_agent": PiHarness,
    "pi-agent": PiHarness,
}


def get_harness(name: Optional[str], **kwargs) -> Optional[BaseHarness]:
    """
    Returns an initialized harness instance by name, or None if name is empty or not found.
    Matching is case-insensitive.
    """
    if not name or not isinstance(name, str):
        return None

    key = name.strip().lower()
    if key in ("none", "default", "raw", ""):
        return None

    harness_cls = HARNESS_REGISTRY.get(key)
    if harness_cls is not None:
        try:
            return harness_cls(**kwargs)
        except TypeError:
            return harness_cls()

    return None
