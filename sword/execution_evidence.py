"""Small, explicitly incomplete summaries of measured execution for the judge."""
import json


def build_execution_evidence(history=(), *, tools_called=0, tools_succeeded=0,
                             test_results=(), execution_results=(), max_chars=4000,
                             max_success=2, max_failed=4):
    if not 512 <= max_chars <= 16000:
        raise ValueError("judge_execution_chars must be between 512 and 16000")
    called = max(0, int(tools_called))
    succeeded = max(0, min(called, int(tools_succeeded)))
    tests = list(test_results)
    evidence = {"tools": {"called": called, "succeeded": succeeded, "failed": called - succeeded},
                "tests": {"total": len(tests), "passed": sum(bool(t.get("success")) for t in tests),
                          "failed": sum(not t.get("success") for t in tests)},
                "samples": [], "omitted": 0, "truncated": False}
    entries = []
    for kind, records in (("test", tests), ("tool", history), ("process", execution_results)):
        for record in records:
            call = record.get("call", record.get("action", {}))
            name = record.get("name") or record.get("tool")
            if not name and isinstance(call, dict):
                name = call.get("payload", {}).get("name") or call.get("type")
            sample = {"kind": kind, "name": str(name or kind)[:100], "success": bool(record.get("success"))}
            if record.get("kind") == "code_io":
                sample["case_type"] = "code_io"
            for key in ("exit_code", "timed_out", "stdout", "stderr", "expected_stdout", "stdin",
                        "output_match", "process_success", "executed", "output_truncated"):
                if key in record:
                    sample[key] = record[key] if key in ("exit_code", "timed_out", "output_match", "process_success",
                                                        "executed", "output_truncated") else str(record[key])[:500]
            if "stdout" not in sample and "stderr" not in sample:
                sample["output"] = str(record.get("output", record.get("message", "")))[:500]
            entries.append(sample)
    # All totals remain exact; samples favor recent failures and dataset tests.
    selected = []
    for success, limit in ((False, max_failed), (True, max_success)):
        candidates = [item for item in entries if item["success"] == success]
        test_samples = [item for item in candidates if item["kind"] == "test"]
        others = [item for item in candidates if item["kind"] != "test"]
        selected.extend((test_samples[-limit:] + others[::-1])[:max(0, limit)] if limit > 0 else [])
    evidence["omitted"] = len(entries) - len(selected)
    for sample in selected:
        evidence["samples"].append(sample)
        if len(json.dumps(evidence, ensure_ascii=False)) > max_chars:
            evidence["samples"].pop()
            evidence["omitted"] += 1
            evidence["truncated"] = True
    evidence["truncated"] = evidence["truncated"] or evidence["omitted"] > 0
    while evidence["samples"] and len(json.dumps(evidence, ensure_ascii=False)) > max_chars:
        evidence["samples"].pop()
        evidence["omitted"] += 1
        evidence["truncated"] = True
    return evidence


def bound_execution_evidence(evidence, max_chars=4000):
    """Keep direct judge callers within the same budget, retaining aggregate counts."""
    if not 512 <= max_chars <= 16000:
        raise ValueError("judge_execution_chars must be between 512 and 16000")
    compact = {"tools": {key: int(evidence.get("tools", {}).get(key, 0)) for key in ("called", "succeeded", "failed")},
               "tests": {key: int(evidence.get("tests", {}).get(key, 0)) for key in ("total", "passed", "failed")},
               "samples": evidence.get("samples", []), "omitted": int(evidence.get("omitted", 0)),
               "truncated": bool(evidence.get("truncated", False))}
    compact["samples"] = list(compact.get("samples", []))
    while compact["samples"] and len(json.dumps(compact, ensure_ascii=False)) > max_chars:
        compact["samples"].pop()
        compact["omitted"] = compact.get("omitted", 0) + 1
        compact["truncated"] = True
    return compact
