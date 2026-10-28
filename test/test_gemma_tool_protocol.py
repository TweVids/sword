"""Render the real Gemma template and execute its native calls through every harness."""
import sys
import json
from pathlib import Path

from jinja2 import Environment
import pytest

from sword.harness import OpenCodeHarness, CodexCliHarness, GeminiCliHarness, PiHarness
from sword.harness.protocol import structured_calls, visible_response


TEMPLATE = Environment().from_string((Path(__file__).parents[1] / "chat_template.jinja").read_text(encoding="utf-8"))


def gemma_call(name, arguments):
    rendered = TEMPLATE.render(bos_token="", enable_thinking=False, effort=None, add_generation_prompt=False,
                               messages=[{"role": "user", "content": "Do the task."},
                                         {"role": "assistant", "content": "", "tool_calls": [
                                             {"function": {"name": name, "arguments": arguments}}]}])
    return rendered.split("<|turn>model\n", 1)[1]


def test_real_template_nested_arguments_and_literal_code_roundtrip():
    arguments = {"path": "state.txt", "name": "argument name", "content": 'print("<think>literal</think>")\nC:\\tmp\\file',
                 "options": {"enabled": True, "nothing": None, "rate": -0.25, "values": [0, "hello", {"x": 1e-5}]}}
    assert structured_calls(gemma_call("write_file", arguments)) == [{"name": "write_file", "args": arguments, "valid_json": True}]


@pytest.mark.parametrize("body", [
    'path:<|"|>state.txt',
    'path:<|"|>state.txt<|"|>',
    'path:<|"|>state.txt<|"|>,path:<|"|>other.txt<|"|>}',
    'path:__import__(<|"|>os<|"|>)}',
    'path:[1,2,]}',
    'path:{x:1},bad:{"name":"python","arguments":{"code":"print(99)"}}',
])
def test_malformed_native_calls_do_not_dispatch_inner_calls(body):
    response = '<|tool_call>call:write_file{' + body + '<tool_call|>'
    calls = structured_calls(response)
    assert calls == [{"name": "write_file", "args": {}, "valid_json": False}]


def test_tagged_native_call_requires_closing_token_and_recovers_on_next_call():
    incomplete = '<|tool_call>call:write_file{path:<|"|>state.txt<|"|>}'
    assert not structured_calls(incomplete)[0]["valid_json"]
    response = incomplete + "\n" + gemma_call("read_file", {"path": "state.txt"})
    assert [call["valid_json"] for call in structured_calls(response)] == [False, True]


@pytest.mark.parametrize("wrapper", ["<think>{}</think>", "<|channel>thought\n{}\n<channel|>",
                                    "```python\n{}\n```", "<|tool_response>{}<tool_response|>"])
def test_native_calls_inside_thoughts_examples_or_tool_responses_are_ignored(wrapper):
    assert structured_calls(wrapper.format(gemma_call("write_file", {"path": "hidden.txt", "content": "hidden"}))) == []


def test_multiple_calls_and_mixed_json_arguments_are_supported():
    response = gemma_call("first", {"a": 1}) + '\ncall:second{a:[true,null,{b:"text"}]}\n' + gemma_call("third", {})
    calls = structured_calls(response)
    assert [call["name"] for call in calls] == ["first", "second", "third"]
    assert calls[1]["args"] == {"a": [True, None, {"b": "text"}]}
    assert all(call["valid_json"] for call in calls)


def test_native_quoted_payload_is_not_also_a_code_block_or_action():
    content = '```python\nraise RuntimeError("must not execute")\n```\nAction: write hidden.txt\nhidden'
    response = gemma_call("write_file", {"path": "document.md", "content": content})
    assert structured_calls(response)[0]["args"]["content"] == content
    assert len(OpenCodeHarness().parse_tools(response)) == 1
    assert len(PiHarness().parse_actions(response)) == 1
    assert len(GeminiCliHarness().parse_function_calls(response)) == 1
    assert CodexCliHarness().extract_code_blocks(response) == []
    assert "<think>literal</think>" in visible_response(gemma_call("python", {"code": 'print("<think>literal</think>")'}))


def test_response_markers_and_calls_inside_quoted_arguments_stay_literal():
    content = "<|tool_response>response:x{value:1}<tool_response|>\n" + gemma_call("python", {"code": "print(99)"})
    # The native template cannot embed its own quote delimiter in a string;
    # use a normal JSON string for this nested literal call.
    response = '<|tool_call>call:write_file{path:"document.md",content:' + json.dumps(content) + '}<tool_call|>'
    calls = structured_calls(response)
    assert len(calls) == 1 and calls[0]["args"]["content"] == content
    assert structured_calls(json.dumps(gemma_call("python", {"code": "print(99)"}))) == []


@pytest.mark.parametrize("cls", [OpenCodeHarness, CodexCliHarness, GeminiCliHarness, PiHarness])
def test_quoted_code_is_written_literally_without_implicit_execution(tmp_path, cls):
    content = '```python\nraise RuntimeError("example only")\n```'
    if cls is CodexCliHarness:
        patch = "*** Begin Patch\n*** Add File: document.md\n" + "\n".join("+" + line for line in content.splitlines()) + "\n*** End Patch"
        response = gemma_call("apply_patch", {"patch": patch})
    else:
        response = gemma_call("write_file", {"path": "document.md", "content": content})
    harness = cls()
    harness.workspace_parent_dir = str(tmp_path)
    turns = []
    def generate(prompt):
        if turns:
            root, = tmp_path.glob("sword_harness_workspace_*")
            assert (root / "document.md").read_text() == content
        turns.append(prompt)
        return response if len(turns) == 1 else "Final Answer: Written."
    result = harness.run("Write a markdown example", generate)
    assert result.tools_called == result.tools_succeeded == 1
    assert result.is_completed and result.tools_failed == 0


def test_native_patch_is_extracted_once_and_fenced_payload_is_not_executed():
    patch = '*** Begin Patch\n*** Add File: document.md\n+```python\n+raise RuntimeError("example")\n+```\n*** End Patch'
    response = gemma_call("apply_patch", {"patch": patch})
    harness = CodexCliHarness()
    assert len(harness.extract_patches(response)) == 1
    assert harness.extract_code_blocks(response) == []


@pytest.mark.parametrize("cls", [OpenCodeHarness, CodexCliHarness, GeminiCliHarness, PiHarness])
def test_native_tools_write_edit_verify_and_finish_in_real_workspace(tmp_path, cls):
    harness = cls()
    harness.workspace_parent_dir = str(tmp_path)
    if cls is CodexCliHarness:
        actions = [gemma_call("apply_patch", {"patch": "*** Begin Patch\n*** Add File: state.txt\n+draft\n*** End Patch"}),
                   gemma_call("apply_patch", {"patch": "*** Begin Patch\n*** Update File: state.txt\n@@\n-draft\n+verified\n*** End Patch"}),
                   gemma_call("exec_command", {"cmd": f'"{sys.executable}" -c "from pathlib import Path; assert Path(\'state.txt\').read_text() == \'verified\'; print(\'CHECKED\')"'})]
    else:
        path_key = "file_path" if cls is GeminiCliHarness else "path"
        old_key, new_key = ("old_string", "new_string") if cls is GeminiCliHarness else (
            ("oldText", "newText") if cls is PiHarness else ("oldString", "newString"))
        actions = [gemma_call("write_file", {path_key: "state.txt", "content": "draft"}),
                   gemma_call("replace" if cls is GeminiCliHarness else "edit", {path_key: "state.txt", old_key: "draft", new_key: "verified"}),
                   gemma_call("python", {"code": "from pathlib import Path\nassert Path('state.txt').read_text() == 'verified'\nprint('CHECKED')"})]
    actions.append("Final Answer: Verified.")
    prompts = []
    def generate(prompt):
        index = len(prompts)
        root, = tmp_path.glob("sword_harness_workspace_*")
        if index:
            assert (root / "state.txt").read_text() == ("draft" if index == 1 else "verified")
        prompts.append(prompt)
        return actions[index]
    result = harness.run("Create, edit and verify state.txt", generate)
    assert result.is_completed and result.turns == 4
    assert result.tools_called == result.tools_succeeded == 3 and result.tools_failed == 0
    assert "CHECKED" in prompts[-1]
    assert result.workspace_removed and not Path(result.workspace_dir).exists()
