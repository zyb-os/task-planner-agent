"""
Regression tests for TaskPlanner._extract_json.

The critical bug: when an LLM returns a clean top-level JSON plan object that
contains nested objects (steps, memory_entries, input_data dicts), the old
reverse-scan heuristic would find the LAST inner `{` first, parse it as a
valid JSON object (e.g. a memory entry or input_data dict), and return it —
producing plan_dict with no "title" or "steps" keys and an "Untitled Workflow"
with 0 steps.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from planner import TaskPlanner


def _planner() -> TaskPlanner:
    return TaskPlanner(orchestrator_base_url="http://localhost:8000")


# ── The exact payload from the bug report ────────────────────────────────────

_LLAMA_RESPONSE = json.dumps({
    "title": "Summary and Email",
    "description": "Create a summary about IRAN war and send it to Rinshad's Gmail.",
    "steps": [
        {
            "name": "Web Search",
            "goal": "Perform a web search for IRAN war information",
            "capability": "browse_web",
            "input_data": {"task": "search for IRAN war news and articles"},
            "confidence": 0.95,
            "execution_mode": "strict",
        },
        {
            "name": "Summary Generation",
            "goal": "Generate a 2-minute summary about IRAN war",
            "capability": "summarize_content",
            "input_data": {
                "data": "{{steps[0].output}}",
                "persona": "formal",
                "delivery_channel": "email",
            },
            "confidence": 0.95,
            "execution_mode": "strict",
        },
        {
            "name": "Email Preparation",
            "goal": "Prepare an email with the summary",
            "capability": "execute_task",
            "input_data": {
                "summary": "{{steps[1].output}}",
                "sender_email": "rinshad.kayilan@gmail.com",
            },
            "confidence": 0.95,
            "execution_mode": "strict",
        },
        {
            "name": "Email Sending",
            "goal": "Send the prepared email",
            "capability": "send_email",
            "input_data": {"to": "rinshad.kayilan@gmail.com", "subject": "Summary about IRAN war"},
            "confidence": 0.95,
            "execution_mode": "strict",
        },
    ],
    "memory_entries": [
        {"category": "Facts", "content": "Rinshad's Gmail account: rinshad.kayilan@gmail.com"}
    ],
})


def test_clean_json_extracts_outer_plan_object() -> None:
    """Regression: clean JSON from the LLM must return the outer plan dict,
    not a nested input_data or memory_entry dict."""
    p = _planner()
    result = p._extract_json(_LLAMA_RESPONSE)
    assert result.get("title") == "Summary and Email", (
        f"Expected 'Summary and Email', got: {result.get('title')!r}\n"
        f"Returned dict keys: {list(result.keys())}"
    )
    assert len(result.get("steps", [])) == 4, (
        f"Expected 4 steps, got {len(result.get('steps', []))}"
    )


def test_clean_json_preserves_all_steps() -> None:
    p = _planner()
    result = p._extract_json(_LLAMA_RESPONSE)
    caps = [s["capability"] for s in result["steps"]]
    assert caps == ["browse_web", "summarize_content", "execute_task", "send_email"]


def test_clean_json_preserves_memory_entries() -> None:
    p = _planner()
    result = p._extract_json(_LLAMA_RESPONSE)
    assert len(result.get("memory_entries", [])) == 1
    assert "Gmail" in result["memory_entries"][0]["content"]


def test_prose_prefix_still_works() -> None:
    """Models that emit a preamble before JSON must still be parsed correctly."""
    p = _planner()
    prose_response = (
        "Here is the workflow plan I've created:\n\n"
        + _LLAMA_RESPONSE
    )
    result = p._extract_json(prose_response)
    assert result.get("title") == "Summary and Email"
    assert len(result.get("steps", [])) == 4


def test_fenced_code_block_still_works() -> None:
    p = _planner()
    fenced = f"```json\n{_LLAMA_RESPONSE}\n```"
    result = p._extract_json(fenced)
    assert result.get("title") == "Summary and Email"


def test_clarification_json_extracted() -> None:
    """Clarification check response (simple bool object) must also parse."""
    p = _planner()
    raw = '{"needs_clarification": false}'
    result = p._extract_json(raw)
    assert result == {"needs_clarification": False}


def test_nested_only_response_raises() -> None:
    """Completely unparseable text must raise ValueError, not silently return {}."""
    import pytest
    p = _planner()
    with pytest.raises(ValueError):
        p._extract_json("no JSON here at all")
