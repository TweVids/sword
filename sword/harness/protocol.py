"""Decode complete tool calls without treating reasoning or malformed calls as answers."""
import json
import math
import re
from bisect import bisect_right


_QUOTED_TEXT = re.compile(r'<\|"\|>.*?<\|"\|>|"(?:\\.|[^"\\])*"', re.DOTALL)


class _GemmaArguments:
    """Decode Gemma objects without rewriting quoted program text or evaluating it."""
    quote = '<|"|>'

    def raw_decode(self, text, start):
        self.text, self.pos = text, start
        return self.value(0), self.pos

    def whitespace(self):
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1

    def string(self):
        if self.text.startswith(self.quote, self.pos):
            start = self.pos + len(self.quote)
            end = self.text.find(self.quote, start)
            if end < 0:
                raise ValueError("Unclosed Gemma string")
            self.pos = end + len(self.quote)
            return self.text[start:end]
        value, consumed = json.JSONDecoder().raw_decode(self.text[self.pos:])
        if not isinstance(value, str):
            raise ValueError("Expected string")
        self.pos += consumed
        return value

    def value(self, depth):
        if depth > 64:
            raise ValueError("Gemma arguments exceed maximum nesting depth")
        self.whitespace()
        if self.pos >= len(self.text):
            raise ValueError("Incomplete Gemma arguments")
        token = self.text[self.pos]
        if self.text.startswith(self.quote, self.pos) or token == '"':
            return self.string()
        if token in "{[":
            closing = "}" if token == "{" else "]"
            result = {} if token == "{" else []
            self.pos += 1
            self.whitespace()
            if self.pos < len(self.text) and self.text[self.pos] == closing:
                self.pos += 1
                return result
            while True:
                self.whitespace()
                if token == "{":
                    if self.text.startswith(self.quote, self.pos) or self.text[self.pos:self.pos + 1] == '"':
                        key = self.string()
                    else:
                        match = re.match(r"[^\s:{}\[\],]+", self.text[self.pos:])
                        if match is None:
                            raise ValueError("Invalid Gemma argument key")
                        key = match.group()
                        self.pos += len(key)
                    self.whitespace()
                    if self.text[self.pos:self.pos + 1] != ":" or key in result:
                        raise ValueError("Missing colon or duplicate Gemma argument key")
                    self.pos += 1
                    result[key] = self.value(depth + 1)
                else:
                    result.append(self.value(depth + 1))
                self.whitespace()
                separator = self.text[self.pos:self.pos + 1]
                self.pos += 1
                if separator == closing:
                    return result
                if separator != ",":
                    raise ValueError("Missing Gemma argument separator or closing bracket")
        match = re.match(r"true|false|null|-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?", self.text[self.pos:])
        if match is None:
            raise ValueError("Invalid Gemma argument value")
        self.pos += len(match.group())
        value = json.loads(match.group())
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Nonfinite Gemma argument")
        return value


def _skip_malformed_arguments(text, start):
    """Do not dispatch nested calls hidden inside an incomplete outer call."""
    position, depth = start, 0
    while position < len(text):
        if text.startswith(_GemmaArguments.quote, position):
            end = text.find(_GemmaArguments.quote, position + len(_GemmaArguments.quote))
            if end < 0:
                return len(text)
            position = end + len(_GemmaArguments.quote)
            continue
        if text[position] == '"':
            try:
                _, consumed = json.JSONDecoder().raw_decode(text[position:])
            except ValueError:
                return len(text)
            position += consumed
            continue
        if text.startswith("<tool_call|>", position):
            return position + len("<tool_call|>")
        if text[position] == "{":
            depth += 1
        elif text[position] == "}":
            depth -= 1
            if depth == 0:
                return position + 1
        position += 1
    return position


def visible_response(text):
    # Markers inside quoted code/tool arguments are literal program text.
    masked = _QUOTED_TEXT.sub(lambda match: " " * len(match.group()), text)
    spans = []
    for pattern, flags in ((r"<think>.*?(?:</think>|$)", re.DOTALL | re.IGNORECASE),
                           (r"<\|channel>\s*thought.*?(?:<channel\|>|<\|end_channel>|$)", re.DOTALL)):
        matches = list(re.finditer(pattern, masked, flags=flags))
        spans.extend(match.span() for match in matches)
        for match in reversed(matches):
            masked = masked[:match.start()] + " " * len(match.group()) + masked[match.end():]
    for start, end in sorted(spans, reverse=True):
        text = text[:start] + text[end:]
    return text


def _scan_calls(text):
    """Parse JSON and Gemma native calls and retain spans for heuristic parsers."""
    text = visible_response(text)
    # Preserve offsets while hiding code examples and runner/tool response text.
    quotes = list(_QUOTED_TEXT.finditer(text))
    quote_starts = [match.start() for match in quotes]
    ignored = re.compile(r"```(?!(?:json|gemma)\b)[^\n]*\n.*?```|<\|tool_response>.*?(?:<tool_response\|>|$)",
                         re.DOTALL | re.IGNORECASE)
    position, hidden = 0, []
    while True:
        match = ignored.search(text, position)
        if match is None:
            break
        quote_index = bisect_right(quote_starts, match.start()) - 1
        if quote_index >= 0 and match.start() < quotes[quote_index].end():
            position = quotes[quote_index].end()
            continue
        hidden.append(match.span())
        position = match.end()
    for start, end in reversed(hidden):
        text = text[:start] + " " * (end - start) + text[end:]
    search_text = _QUOTED_TEXT.sub(lambda match: " " * len(match.group()), text)
    decoder = json.JSONDecoder()
    calls = []
    spans = []

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
        match = re.search(r"call:([A-Za-z0-9_.-]+)\s*(?=\{)|[\[{]", search_text[position:])
        if match is None:
            break
        start = position + match.start()
        forced_name = match.group(1)
        json_start = position + match.end() if forced_name else start
        tagged = bool(forced_name and re.search(r"<\|tool_call>\s*$", text[max(0, start - 40):start]))
        span_start = text.rfind("<|tool_call>", max(0, start - 40), start) if tagged else start
        try:
            data, consumed = decoder.raw_decode(text[json_start:])
        except ValueError:
            if forced_name:
                try:
                    data, end = _GemmaArguments().raw_decode(text, json_start)
                    consumed = end - json_start
                except (ValueError, IndexError, RecursionError):
                    position = _skip_malformed_arguments(text, json_start)
                    calls.append({"name": forced_name, "args": {}, "valid_json": False})
                    spans.append((span_start, position))
                    continue
            else:
                position = json_start + 1
                continue
        position = json_start + consumed
        if tagged:
            closing = re.match(r"\s*<tool_call\|>", text[position:])
            if closing is None:
                calls.append({"name": forced_name, "args": {}, "valid_json": False})
                spans.append((span_start, position))
                continue
            position += closing.end()
        previous_count = len(calls)
        if forced_name:
            if isinstance(data, dict):
                normalize({"name": forced_name, "arguments": data})
            else:
                calls.append({"name": forced_name, "args": {}, "valid_json": False})
        else:
            normalize(data)
        if len(calls) > previous_count:
            spans.append((span_start, position))
    return calls, spans


def structured_calls(text):
    """Accept JSON or Gemma call:name{key:<|\"|>value<|\"|>} tool arguments."""
    return _scan_calls(text)[0]


def without_structured_calls(text):
    """Keep code/patch heuristics from also executing text inside tool arguments."""
    text = visible_response(text)
    _, spans = _scan_calls(text)
    for start, end in reversed(spans):
        text = text[:start] + " " * (end - start) + text[end:]
    return text
