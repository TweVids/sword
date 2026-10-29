"""Model defaults shared by setup, diagnostics, and GRPO training."""

DEFAULT_INFERENCE_MODEL = "Qwen/Qwen3.5-0.8B"
DEFAULT_GRPO_MODEL = "google/gemma-4-31B-it"
DEFAULT_B_MODEL = "qwen3.8:27b"
DEFAULT_B_MODEL_FALLBACK = "qwen3.5:9b"

# Shared budgets for diagnostic and ordinary agent training. Explicit caller
# settings still override these defaults.
DEFAULT_GRPO_CONTEXT = 12_288
DEFAULT_GRPO_MAX_NEW_TOKENS = 2_048
DEFAULT_MOCK_MAX_NEW_TOKENS = 64
DEFAULT_B_CONTEXT = 32_768
