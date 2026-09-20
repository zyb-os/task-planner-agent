from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from planner import TaskPlanner


def _agent(name: str, caps: list[dict]) -> dict:
    return {"name": name, "disabled": False, "capabilities": caps}


def test_compact_capability_selection_limits_output() -> None:
    planner = TaskPlanner(orchestrator_base_url="http://localhost:8000")
    agents = [
        _agent(
            "serper-search-agent",
            [{"name": f"serper_search_{i}", "input_schema": {"properties": {"query": {"type": "string"}}}}
             for i in range(8)],
        ),
        _agent(
            "messaging-agent",
            [{"name": "send_slack_message", "input_schema": {"properties": {"channel": {"type": "string"}, "text": {"type": "string"}}, "required": ["channel", "text"]}}],
        ),
        _agent(
            "scheduler-agent",
            [{"name": "schedule_task", "input_schema": {"properties": {"scheduled_at": {"type": "string"}}, "required": ["scheduled_at"]}}],
        ),
    ]
    text = planner._format_capabilities(agents, goal="plan vacation and remind me later", compact=True)
    # Keep prompt compact while still preserving messaging + scheduling options.
    assert text.count("(agent: ") <= 10
    assert "send_slack_message" in text
    assert "schedule_task" in text


def test_compact_memory_context_truncates_large_markdown() -> None:
    planner = TaskPlanner(orchestrator_base_url="http://localhost:8000")
    memory = "\n".join([f"- line {i} abcdefghijklmnopqrstuvwxyz" for i in range(200)])
    compact = planner._compact_memory_context(memory, max_chars=220)
    assert len(compact) <= 240
    assert "truncated" in compact


# ── Progressive-disclosure / token-reduction strategies ─────────────────────

def _big_catalogue() -> list[dict]:
    """A catalogue larger than _COMPACT_CAP_LIMIT so tiering kicks in."""
    return [
        _agent(
            "serper-search-agent",
            [
                {
                    "name": f"serper_search_{i}",
                    "description": "Use this capability to search the web for results",
                    "input_schema": {
                        "properties": {"query": {"type": "string"}, "count": {"type": "integer"}},
                        "required": ["query"],
                    },
                }
                for i in range(15)
            ],
        ),
        _agent(
            "messaging-agent",
            [
                {
                    "name": "send_slack_message",
                    "description": "Allows you to post a message to a channel",
                    "input_schema": {
                        "properties": {"channel": {"type": "string"}, "text": {"type": "string"}},
                        "required": ["channel", "text"],
                    },
                }
            ],
        ),
        _agent(
            "scheduler-agent",
            [
                {
                    "name": "schedule_task",
                    "description": "Schedule a task to run later",
                    "input_schema": {
                        "properties": {"scheduled_at": {"type": "string"}},
                        "required": ["scheduled_at"],
                    },
                }
            ],
        ),
    ]


def test_tiered_format_keeps_every_capability_visible() -> None:
    """Progressive disclosure must NOT drop any capability — the long tail is
    listed name-only, but every capability name still appears."""
    planner = TaskPlanner(orchestrator_base_url="http://localhost:8000")
    text = planner._format_capabilities(_big_catalogue(), goal="search the web", compact=True)

    for i in range(15):
        assert f"serper_search_{i}" in text, f"serper_search_{i} was dropped"
    assert "send_slack_message" in text
    assert "schedule_task" in text
    # The long tail is disclosed under the names-only section.
    assert "More capabilities available" in text


def test_tiered_format_is_smaller_than_verbose() -> None:
    """The whole catalogue, compacted, should be materially smaller than the
    verbose rendering of the same caps (char count as a token proxy)."""
    planner = TaskPlanner(orchestrator_base_url="http://localhost:8000")
    agents = _big_catalogue()
    compact = planner._format_capabilities(agents, goal="search the web", compact=True)
    verbose = planner._format_capabilities(agents, compact=False)
    assert len(compact) < len(verbose) * 0.6


def test_compact_drops_free_cost_and_string_types() -> None:
    """Drop-inferred: 'free' never printed, string types omitted, required
    fields carry '!', optional collapse to a count."""
    planner = TaskPlanner(orchestrator_base_url="http://localhost:8000")
    agents = [
        _agent(
            "messaging-agent",
            [
                {
                    "name": "send_slack_message",
                    "description": "Post a message",
                    "input_schema": {
                        "properties": {
                            "channel": {"type": "string"},
                            "text": {"type": "string"},
                            "thread_ts": {"type": "string"},
                        },
                        "required": ["channel", "text"],
                    },
                }
            ],
        ),
    ]
    text = planner._format_capabilities(agents, goal="send a slack message", compact=True)
    assert "free" not in text                 # free cost omitted
    assert "channel!" in text and "text!" in text  # required marked with !
    assert "(str" not in text                 # string type code omitted
    assert "+1opt" in text                    # optional collapsed to a count


def test_agent_level_factoring_hoists_shared_path_constraint() -> None:
    """A path constraint shared by every capability in an agent appears once in
    the header, not repeated per capability."""
    planner = TaskPlanner(orchestrator_base_url="http://localhost:8000")
    caps = [
        {
            "agent_name": "filesystem-agent",
            "capability_name": "read_file",
            "description": "Read a file",
            "required_fields": ["path"],
            "optional_fields": [],
            "properties": {"path": {"type": "string"}},
            "cost_usd": None,
            "path_constraint": ["/workspace"],
        },
        {
            "agent_name": "filesystem-agent",
            "capability_name": "write_file",
            "description": "Write a file",
            "required_fields": ["path", "content"],
            "optional_fields": [],
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "cost_usd": None,
            "path_constraint": ["/workspace"],
        },
    ]
    text = planner._format_caps_compact(caps)
    # Constraint hoisted to header exactly once.
    assert text.count("paths⊂/workspace") == 1
    assert "[filesystem-agent | paths⊂/workspace]" in text
