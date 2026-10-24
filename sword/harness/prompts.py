"""
Harness System Prompts Loader:
Loads authentic system prompts and tool definitions directly from cloned third_party repositories:
- OpenCode (third_party/opencode)
- Pi (third_party/pi)
- Codex CLI (third_party/codex)
- Gemini CLI (third_party/gemini_cli)
Automatically clones any missing repository shallowly on first load.
"""

import os
import re
from typing import Dict, List, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
THIRD_PARTY_DIR = os.path.join(REPO_ROOT, "third_party")

HARNESS_GIT_REPOS: Dict[str, str] = {
    "opencode": "https://github.com/anomalyco/opencode.git",
    "pi": "https://github.com/earendil-works/pi.git",
    "codex": "https://github.com/openai/codex.git",
    "gemini_cli": "https://github.com/google-gemini/gemini-cli.git",
}


def ensure_harness_repo(harness_key: str) -> bool:
    """Clones the authentic harness repo shallowly into third_party if not present."""
    import subprocess
    repo_url = HARNESS_GIT_REPOS.get(harness_key)
    if not repo_url:
        return False
    dest_dir = os.path.join(THIRD_PARTY_DIR, harness_key)
    if os.path.isdir(dest_dir):
        return True
    if os.environ.get("SWORD_AUTO_CLONE_HARNESS", "1") == "0":
        return False
    os.makedirs(THIRD_PARTY_DIR, exist_ok=True)
    try:
        print(f"[HarnessLoader] Cloning authentic harness {harness_key} from {repo_url}...")
        subprocess.run(["git", "clone", "--depth", "1", repo_url, dest_dir], check=True, timeout=120)
        return os.path.isdir(dest_dir)
    except Exception as e:
        print(f"[HarnessLoader] Note: unable to clone {harness_key}: {e}. Fallback prompts will be used.")
        return False


def load_opencode_prompt() -> str:
    """Loads authentic OpenCode system prompt and tool plugin schemas from cloned repository."""
    ensure_harness_repo("opencode")
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
                "- Use the edit tool for targeted changes to existing text files with oldString and newString."
            )
            base_prompt = content.replace("${OPENCODE_TOOL_GUIDANCE}", guidance)
        except Exception:
            base_prompt = "You are an AI agent running in OpenCode, a coding agent harness."
    else:
        base_prompt = "You are an AI agent running in OpenCode, a coding agent harness."

    # Extract tool schemas from OpenCode plugin definitions
    plugin_dir = os.path.join(THIRD_PARTY_DIR, "opencode", "packages", "core", "src", "tool", "plugin")
    plugin_summaries: List[str] = []
    for tool_file, schema_hint in (
        ("shell.ts", '{"action": "shell", "parameters": {"command": "...", "workdir": "...", "timeout": 120000}}'),
        ("write.ts", '{"action": "write", "parameters": {"path": "...", "content": "..."}}'),
        ("edit.ts", '{"action": "edit", "parameters": {"path": "...", "oldString": "...", "newString": "...", "replaceAll": false}}'),
        ("read.ts", '{"action": "read", "parameters": {"path": "...", "offset": 1, "limit": 2000}}'),
    ):
        tpath = os.path.join(plugin_dir, tool_file)
        desc = ""
        if os.path.isfile(tpath):
            try:
                with open(tpath, "r", encoding="utf-8") as tf:
                    tsrc = tf.read()
                m_desc = re.search(r'editor\.add\(\{.*?description:\s*"([^"]+)"', tsrc, re.DOTALL)
                if not m_desc:
                    m_desc = re.search(r'description:\s*"([^"]+)"', tsrc)
                if m_desc:
                    desc = m_desc.group(1).split("\\n")[0]
            except Exception:
                pass
        tname = tool_file.replace(".ts", "")
        if desc:
            plugin_summaries.append(f"- {tname}: {desc}\n  Call: <tool_call>{schema_hint}</tool_call>")
        else:
            plugin_summaries.append(f"- {tname}: <tool_call>{schema_hint}</tool_call>")

    tool_instructions = (
        "\n\n# OpenCode Tool Definitions:\n"
        + "\n".join(plugin_summaries)
        + "\n- Code execution blocks: ```bash\ncommand\n``` or ```python\ncode\n```"
    )
    return base_prompt + tool_instructions


def load_codex_prompt() -> str:
    """Loads authentic Codex CLI GPT-5 prompt, apply_patch grammar, and exec_command spec from cloned repository."""
    ensure_harness_repo("codex")
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

    lark_path = os.path.join(
        THIRD_PARTY_DIR,
        "codex",
        "codex-rs",
        "core",
        "assets",
        "tools",
        "apply_patch.lark",
    )
    lark_grammar = ""
    if os.path.isfile(lark_path):
        try:
            with open(lark_path, "r", encoding="utf-8") as lf:
                lark_grammar = lf.read().strip()
        except Exception:
            pass

    tool_instructions = (
        "\n\n# Codex CLI Tool Specifications:\n"
        "1. `exec_command` / `shell`: Execute a shell command with `{\"cmd\": \"...\", \"workdir\": \"...\"}` or inside ```bash\n...\n``` or ```python\n...\n``` blocks.\n"
        "2. `apply_patch`: Edit, add, or delete files using the freeform patch grammar:\n"
        "*** Begin Patch\n"
        "*** Update File: path/to/file.py\n"
        "@@\n"
        "-old line\n"
        "+new line\n"
        "*** End Patch"
    )
    if lark_grammar:
        tool_instructions += f"\n\nGrammar (`apply_patch.lark`):\n{lark_grammar}"
    return base_prompt + tool_instructions


def load_pi_prompt() -> str:
    """Loads authentic Pi system prompt and tool contributions from packages/coding-agent/src/core/system-prompt.ts."""
    ensure_harness_repo("pi")
    sys_prompt_ts = os.path.join(
        THIRD_PARTY_DIR,
        "pi",
        "packages",
        "coding-agent",
        "src",
        "core",
        "system-prompt.ts",
    )
    tools_dir = os.path.join(
        THIRD_PARTY_DIR,
        "pi",
        "packages",
        "coding-agent",
        "src",
        "core",
        "tools",
    )

    preamble = (
        "You are an expert coding assistant operating inside pi, a coding agent harness. "
        "You help users by reading files, executing commands, editing code, and writing new files."
    )
    rules: List[str] = ["Use bash for file operations like ls, rg, find"]

    if os.path.isfile(sys_prompt_ts):
        try:
            with open(sys_prompt_ts, "r", encoding="utf-8") as f:
                ts_content = f.read()
            m_pre = re.search(r'promptSections\.preamble\s*=\s*"([^"]+)"', ts_content)
            if m_pre:
                preamble = m_pre.group(1)
        except Exception:
            pass

    tool_lines: List[str] = []
    for tname in ("read", "bash", "edit", "write"):
        tfile = os.path.join(tools_dir, f"{tname}.ts")
        snippet = ""
        if os.path.isfile(tfile):
            try:
                with open(tfile, "r", encoding="utf-8") as tf:
                    tcontent = tf.read()
                m_snip = re.search(r'snippet:\s*"([^"]+)"', tcontent)
                if m_snip:
                    snippet = m_snip.group(1)
                m_guide_block = re.search(r"guidelines:\s*\[(.*?)\]", tcontent, re.DOTALL)
                if m_guide_block:
                    for g in re.findall(r'"([^"]+)"', m_guide_block.group(1)):
                        if g not in rules:
                            rules.append(g)
            except Exception:
                pass
        if snippet:
            tool_lines.append(f"- {tname}: {snippet}")

    rules.extend([
        "Be concise in your responses",
        "Show file paths clearly when working with files",
    ])

    tools_section = "\n".join(tool_lines) if tool_lines else (
        "- read: Read file contents\n"
        "- bash: Execute bash commands (ls, grep, find, etc.)\n"
        "- edit: Make precise file edits with exact text replacement, including multiple disjoint edits in one call\n"
        "- write: Create or overwrite files"
    )
    rules_section = "\n".join(f"- {r}" for r in rules)

    schema_instructions = (
        "\n\n<tool_schemas>\n"
        "- read: {\"action\": \"read\", \"parameters\": {\"path\": \"...\", \"offset\": 1, \"limit\": 200}} or `Action: read <path>`\n"
        "- bash: {\"action\": \"bash\", \"parameters\": {\"command\": \"...\"}} or `Action: bash <command>` or ```bash ... ```\n"
        "- edit: {\"action\": \"edit\", \"parameters\": {\"path\": \"...\", \"edits\": [{\"oldText\": \"...\", \"newText\": \"...\"}]}}\n"
        "- write: {\"action\": \"write\", \"parameters\": {\"path\": \"...\", \"content\": \"...\"}} or `Action: write <path>\\n<content>`\n"
        "</tool_schemas>"
    )

    return (
        f"{preamble}\n\n"
        f"<tools>\n{tools_section}\n\nIn addition to the tools above, you may have access to other custom tools depending on the project.\n</tools>\n\n"
        f"<rules>\n{rules_section}\n</rules>"
        f"{schema_instructions}"
    )


def load_gemini_prompt() -> str:
    """Loads authentic Gemini CLI system prompt from packages/core/src/prompts/snippets.ts and base-declarations.ts."""
    ensure_harness_repo("gemini_cli")
    snippets_path = os.path.join(
        THIRD_PARTY_DIR,
        "gemini_cli",
        "packages",
        "core",
        "src",
        "prompts",
        "snippets.ts",
    )
    decls_path = os.path.join(
        THIRD_PARTY_DIR,
        "gemini_cli",
        "packages",
        "core",
        "src",
        "tools",
        "definitions",
        "base-declarations.ts",
    )

    tool_consts: Dict[str, str] = {
        "GLOB_TOOL_NAME": "glob",
        "GREP_TOOL_NAME": "grep_search",
        "LS_TOOL_NAME": "list_directory",
        "READ_FILE_TOOL_NAME": "read_file",
        "SHELL_TOOL_NAME": "run_shell_command",
        "WRITE_FILE_TOOL_NAME": "write_file",
        "EDIT_TOOL_NAME": "replace",
        "EDIT_PARAM_OLD_STRING": "old_string",
        "GREP_PARAM_TOTAL_MAX_MATCHES": "total_max_matches",
        "GREP_PARAM_INCLUDE_PATTERN": "include_pattern",
        "GREP_PARAM_EXCLUDE_PATTERN": "exclude_pattern",
        "GREP_PARAM_CONTEXT": "context",
        "GREP_PARAM_BEFORE": "before",
        "GREP_PARAM_AFTER": "after",
        "READ_FILE_PARAM_START_LINE": "start_line",
        "READ_FILE_PARAM_END_LINE": "end_line",
    }

    if os.path.isfile(decls_path):
        try:
            with open(decls_path, "r", encoding="utf-8") as df:
                dcontent = df.read()
            for k, v in re.findall(r"export const ([A-Z0-9_]+)\s*=\s*'([^']+)';", dcontent):
                tool_consts[k] = v
        except Exception:
            pass

    extracted_sections: List[str] = [
        "You are Gemini CLI, an autonomous CLI agent specializing in software engineering tasks. You are currently operating in **Default** mode. Your primary goal is to help users safely and effectively."
    ]

    if os.path.isfile(snippets_path):
        try:
            with open(snippets_path, "r", encoding="utf-8") as sf:
                scontent = sf.read()

            # Extract Core Mandates template from renderCoreMandates
            m_mandates = re.search(r"export function renderCoreMandates.*?return `(.*?)`\.trim\(\);", scontent, re.DOTALL)
            if m_mandates:
                mandates_text = m_mandates.group(1).strip()
                mandates_text = mandates_text.replace("${formattedFilenames}", "`GEMINI.md`")
                for k, v in tool_consts.items():
                    mandates_text = mandates_text.replace(f"${{{k}}}", v)
                # Strip remaining dynamic TS template expressions cleanly
                mandates_text = re.sub(r"\$\{[^}]+\}", "", mandates_text)
                extracted_sections.append(mandates_text)

            # Extract Primary Workflows template from renderPrimaryWorkflows
            m_workflows = re.search(r"export function renderPrimaryWorkflows.*?return `(.*?)`\.trim\(\);", scontent, re.DOTALL)
            if m_workflows:
                wf_text = m_workflows.group(1).strip()
                for k, v in tool_consts.items():
                    wf_text = wf_text.replace(f"${{formatToolName({k})}}", f"`{v}`")
                    wf_text = wf_text.replace(f"${{{k}}}", v)
                wf_text = re.sub(r"\$\{[^}]+\}", "", wf_text)
                extracted_sections.append(wf_text)
        except Exception:
            pass

    tool_protocol = (
        "\n\n# Gemini CLI Tool Declarations:\n"
        f"- `call:{tool_consts['SHELL_TOOL_NAME']}{{\"command\": \"<shell command>\"}}`\n"
        f"- `call:{tool_consts['READ_FILE_TOOL_NAME']}{{\"file_path\": \"<path>\", \"start_line\": 1, \"end_line\": 200}}`\n"
        f"- `call:{tool_consts['WRITE_FILE_TOOL_NAME']}{{\"file_path\": \"<path>\", \"content\": \"<text>\"}}`\n"
        f"- `call:{tool_consts['EDIT_TOOL_NAME']}{{\"file_path\": \"<path>\", \"old_string\": \"<old>\", \"new_string\": \"<new>\"}}`\n"
        f"- `call:{tool_consts['LS_TOOL_NAME']}{{\"dir_path\": \".\"}}`\n"
        f"- `call:{tool_consts['GREP_TOOL_NAME']}{{\"pattern\": \"<regex>\"}}`\n"
        "- `call:calculator{\"expression\": \"<math expr>\"}`\n"
        "- `call:python{\"code\": \"<python code>\"}`"
    )

    return "\n\n".join(extracted_sections) + tool_protocol


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


def get_harness_system_prompt(name: Optional[str]) -> str:
    """Retrieves the authentic system prompt for a named harness."""
    if not name or not isinstance(name, str):
        return ""
    key = name.strip().lower()
    fn = PROMPT_REGISTRY.get(key)
    if fn:
        return fn()
    return ""
