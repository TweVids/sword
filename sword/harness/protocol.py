"""Decode complete tool calls without treating reasoning or malformed calls as answers."""
import json
import re


def visible_response(text):
    text = re.sub(r"<think>.*?(?:</think>|$)", "", text, flags=re.DOTALL | re.IGNORECASE)
    return re.sub(r"<\|channel>thought.*?(?:<channel\||<\|end_channel>|$)", "", text, flags=re.DOTALL)


def structured_calls(text):
    """Accept tagged, fenced, bare, and function-style JSON, including nested args."""
    text = visible_response(text)
    text = re.sub(r"```(?!json\b)[^\n]*\n.*?```", "", text, flags=re.DOTALL | re.IGNORECASE)
    decoder = json.JSONDecoder()
    calls = []

    def normalize(data, forced_name=None):
        if isinstance(data, list):
            for item in data:
                normalize(item)
            return
        if not isinstance(data, dict):
            return
        if isinstance(data.get("tool_calls"), list):
            normalize(data["tool_calls"])
            return
        function = data.get("function")
        if isinstance(function, dict):
            data = function
        name = forced_name or data.get("action") or data.get("tool") or data.get("name")
        if not name and isinstance(data.get("cmd"), str):
            name = "exec_command"
        if not isinstance(name, str) or not name:
            return
        args = next((data[key] for key in ("parameters", "arguments", "args", "input") if key in data), None)
        if args is None:
            args = {key: value for key, value in data.items() if key not in ("action", "tool", "name", "type", "id")}
        if isinstance(args, str) and name != "apply_patch":
            try:
                args = json.loads(args)
            except ValueError:
                pass
        calls.append({"name": name, "args": args, "valid_json": isinstance(args, dict) or name == "apply_patch"})

    position = 0
    while position < len(text):
        match = re.search(r"call:([A-Za-z0-9_]+)\s*(?=\{)|[\[{]", text[position:])
        if match is None:
            break
        start = position + match.start()
        forced_name = match.group(1)
        json_start = position + match.end() if forced_name else start
        try:
            data, consumed = decoder.raw_decode(text[json_start:])
        except ValueError:
            if forced_name:
                calls.append({"name": forced_name, "args": {}, "valid_json": False})
            position = json_start + 1
            continue
        normalize(data, forced_name)
        position = json_start + consumed
    return calls
