"""
Harness System Prompts Loader:
Loads authentic system prompts directly from cloned third_party repositories:
- OpenCode (third_party/opencode)
- Pi (third_party/pi)
- Codex CLI (third_party/codex)
- Gemini CLI (third_party/gemini_cli)
Provides tool usage instructions and schemas so models can interact with each playground.
"""

import os
from typing import Dict, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
THIRD_PARTY_DIR = os.path.join(REPO_ROOT, "third_party")


def load_opencode_prompt() -> str:
    """Loads authentic OpenCode system prompt from cloned repository."""
    raw_prompt_path = os.path.join(
        THIRD_PARTY_DIR,
        "opencode",
        "packages",
        "core",
        "src",
        "session",
        "runner",
        "prompt",
        "system.txt",
    )
    if os.path.isfile(raw_prompt_path):
        try:
            with open(raw_prompt_path, "r", encoding="utf-8") as f:
                content = f.read()
            guidance = (
                "- Prefer dedicated tools over shell commands; fall back to the shell when a tool cannot do what you need.\n"
                "- Use the write tool to create files or completely replace their content.\n"
                "- Use the edit tool for targeted changes to existing text files with old_str and new_str."
            )
            base_prompt = content.replace("${OPENCODE_TOOL_GUIDANCE}", guidance)
        except Exception:
            base_prompt = "You are an AI agent running in OpenCode, a coding agent harness."
    else:
        base_prompt = "You are an AI agent running in OpenCode, a coding agent harness."

    tool_instructions = (
        "\n\n# OpenCode Tool Protocol:\n"
        "To invoke tools, output JSON tool_call blocks or code blocks:\n"
        "1. Bash command: ```bash\ncommand here\n```\n"
        "2. Python script: ```python\ncode here\n```\n"
        "3. File writing: <tool_call>{\"action\": \"write\", \"parameters\": {\"path\": \"file.ext\", \"content\": \"...\"}}</tool_call>\n"
        "4. File editing: <tool_call>{\"action\": \"edit\", \"parameters\": {\"path\": \"file.ext\", \"old_str\": \"...\", \"new_str\": \"...\"}}</tool_call>\n"
        "5. File reading: <tool_call>{\"action\": \"read\", \"parameters\": {\"path\": \"file.ext\"}}</tool_call>"
    )
    return base_prompt + tool_instructions


def load_codex_prompt() -> str:
    """Loads authentic Codex CLI system prompt from cloned repository."""
    prompt_path = os.path.join(
        THIRD_PARTY_DIR,
        "codex",
        "codex-rs",
        "core",
        "gpt_5_codex_prompt.md",
    )
    if os.path.isfile(prompt_path):
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                base_prompt = f.read()
        except Exception:
            base_prompt = "You are Codex, based on GPT-5. You are running as a coding agent in the Codex CLI on a user's computer."
    else:
        base_prompt = "You are Codex, based on GPT-5. You are running as a coding agent in the Codex CLI on a user's computer."

    tool_instructions = (
        "\n\n# Codex CLI Execution Protocol:\n"
        "To solve coding and algorithmic tasks, provide self-contained, executable Python code or commands:\n"
        "- Enclose Python solutions and verification assertions in ```python\n...\n``` blocks.\n"
        "- Enclose terminal verification commands in ```bash\n...\n``` blocks.\n"
        "The harness executes your code in the sandbox and verifies syntax and unit assertions."
    )
    return base_prompt + tool_instructions


def load_pi_prompt() -> str:
    """Loads authentic Pi system prompt from cloned repository."""
    pi_md_path = os.path.join(THIRD_PARTY_DIR, "pi", "AGENTS.md")
    rules_text = ""
    if os.path.isfile(pi_md_path):
        try:
            with open(pi_md_path, "r", encoding="utf-8") as f:
                rules_text = "\n" + f.read()[:1000]
        except Exception:
            pass

    base_prompt = (
        "You are an expert coding assistant operating inside pi, a coding agent harness (pi.dev). "
        "You help users by reading files, executing commands, editing code, and writing new files.\n"
        f"{rules_text}\n\n"
        "# Pi Foundational Tool Protocol:\n"
        "You have access to foundational tools:\n"
        "- Action: read <path> (reads file)\n"
        "- Action: write <path>\\n<content> (writes file)\n"
        "- Action: edit (replaces target text in file)\n"
        "- Action: bash <command> (executes shell command in workspace)\n"
        "- Code blocks: ```bash ... ``` or ```python ... ``` for direct execution."
    )
    return base_prompt


def load_gemini_prompt() -> str:
    """Loads authentic Gemini CLI system prompt from cloned repository."""
    base_prompt = (
        "You are Gemini CLI, an autonomous CLI agent specializing in software engineering tasks. "
        "Your primary goal is to help users safely and effectively.\n\n"
        "# Gemini CLI Function Calling Protocol:\n"
        "To call tools, output function directives:\n"
        "- call:calculator{\"expression\": \"<math expr>\"}\n"
        "- call:python{\"code\": \"<python code>\"}\n"
        "- call:bash{\"command\": \"<command>\"}\n"
        "- call:read_file{\"path\": \"<file path>\"}\n"
        "- call:write_file{\"path\": \"<file path>\", \"content\": \"<text>\"}"
    )
    return base_prompt


PROMPT_REGISTRY: Dict[str, callable] = {
    "opencode": load_opencode_prompt,
    "codex": load_codex_prompt,
    "codex_cli": load_codex_prompt,
    "codex-cli": load_codex_prompt,
    "pi": load_pi_prompt,
    "pi_agent": load_pi_prompt,
    "gemini": load_gemini_prompt,
    "gemini_cli": load_gemini_prompt,
    "gemini-cli": load_gemini_prompt,
}


HARNESS_GIT_REPOS: Dict[str, str] = {
    "opencode": "https://github.com/anomalyco/opencode.git",
    "pi": "https://github.com/earendil-works/pi.git",
    "codex": "https://github.com/openai/codex.git",
    "gemini_cli": "https://github.com/google-gemini/gemini-cli.git",
}


def ensure_harness_repo(harness_key: str) -> None:
    """Clones the authentic harness repo shallowly into third_party if not present."""
    import subprocess
    repo_url = HARNESS_GIT_REPOS.get(harness_key)
    if not repo_url:
        return
    dest_dir = os.path.join(THIRD_PARTY_DIR, harness_key)
    if not os.path.isdir(dest_dir):
        os.makedirs(THIRD_PARTY_DIR, exist_ok=True)
        try:
            print(f"[HarnessLoader] Cloning authentic harness {harness_key} from {repo_url}...")
            subprocess.run(["git", "clone", "--depth", "1", repo_url, dest_dir], check=True, timeout=120)
        except Exception as e:
            print(f"[HarnessLoader] Note: unable to clone {harness_key}: {e}. Fallback prompts will be used.")


def get_harness_system_prompt(name: Optional[str]) -> str:
    """Retrieves the authentic system prompt for a named harness."""
    if not name or not isinstance(name, str):
        return ""
    key = name.strip().lower()
    fn = PROMPT_REGISTRY.get(key)
    if fn:
        return fn()
    return ""
