from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from orchestrator_client import StepRefError, _resolve_step_refs


# ── Happy paths (behaviour that must not regress) ─────────────────────────────


def test_whole_output_passthrough_returns_dict() -> None:
    outputs = [{"summary": "hello", "count": 2}]
    assert _resolve_step_refs("{{steps[0].output}}", outputs) == {
        "summary": "hello",
        "count": 2,
    }


def test_dotted_path_and_array_index() -> None:
    outputs = [{"results": [{"url": "https://a"}, {"url": "https://b"}]}]
    assert _resolve_step_refs("{{steps[0].output.results[1].url}}", outputs) == "https://b"


def test_interpolation_inside_larger_string() -> None:
    outputs = [{"name": "ZybOS"}]
    assert _resolve_step_refs("Hello {{steps[0].output.name}}!", outputs) == "Hello ZybOS!"


def test_nested_structures_are_resolved() -> None:
    outputs = [{"id": 7}]
    value = {"a": ["{{steps[0].output.id}}"], "b": {"c": "{{steps[0].output.id}}"}}
    assert _resolve_step_refs(value, outputs) == {"a": [7], "b": {"c": 7}}


def test_present_but_null_value_is_not_an_error() -> None:
    """A key that exists with value None is a real value, not a missing field."""
    outputs = [{"voice_id": None}]
    assert _resolve_step_refs("{{steps[0].output.voice_id}}", outputs) is None


def test_plain_strings_untouched() -> None:
    assert _resolve_step_refs("no refs here", [{}]) == "no refs here"


# ── Fail-fast paths (the actual fix) ──────────────────────────────────────────


def test_missing_field_raises_and_blames_producing_step() -> None:
    """Regression: browse_web returns {summary}, plans referenced .headlines."""
    outputs = [{"summary": "..."}]
    with pytest.raises(StepRefError) as ei:
        _resolve_step_refs("{{steps[0].output.headlines}}", outputs)
    assert ei.value.producing_step == 0
    msg = str(ei.value)
    assert "headlines" in msg
    assert "summary" in msg  # available keys are surfaced for debugging


def test_literal_template_no_longer_leaks_downstream() -> None:
    """Regression for the logged failure:
    "No available agent for capability '{{steps[3].output.capability}}'"."""
    outputs = [{}, {}, {}, {"other": 1}]
    with pytest.raises(StepRefError):
        _resolve_step_refs("{{steps[3].output.capability}}", outputs)


def test_index_out_of_range_raises() -> None:
    with pytest.raises(StepRefError, match="only 1 step"):
        _resolve_step_refs("{{steps[4].output.x}}", [{"x": 1}])


def test_step_with_no_output_raises() -> None:
    with pytest.raises(StepRefError, match="produced no output"):
        _resolve_step_refs("{{steps[0].output.x}}", [None])


def test_forward_reference_rejected() -> None:
    outputs = [{"a": 1}, None, None]
    with pytest.raises(StepRefError, match="runs later"):
        _resolve_step_refs("{{steps[2].output.a}}", outputs, current_index=1)


def test_self_reference_rejected() -> None:
    outputs = [{"a": 1}, None]
    with pytest.raises(StepRefError, match="itself"):
        _resolve_step_refs("{{steps[1].output.a}}", outputs, current_index=1)


def test_traversal_through_non_dict_reports_actual_shape() -> None:
    outputs = [{"summary": "a plain string"}]
    with pytest.raises(StepRefError) as ei:
        _resolve_step_refs("{{steps[0].output.summary.title}}", outputs)
    assert "str" in str(ei.value)


def test_failure_inside_nested_structure_propagates() -> None:
    outputs = [{"summary": "x"}]
    with pytest.raises(StepRefError):
        _resolve_step_refs({"body": {"t": ["{{steps[0].output.nope}}"]}}, outputs)


# ── Lenient mode (emergent-runner hints) ──────────────────────────────────────


def test_lenient_mode_leaves_unresolvable_refs_untouched() -> None:
    outputs = [{"summary": "x"}]
    ref = "{{steps[0].output.headlines}}"
    assert _resolve_step_refs(ref, outputs, strict=False) == ref


def test_lenient_mode_still_resolves_what_it_can() -> None:
    outputs = [{"summary": "x"}]
    value = {"good": "{{steps[0].output.summary}}", "bad": "{{steps[0].output.nope}}"}
    got = _resolve_step_refs(value, outputs, strict=False)
    assert got["good"] == "x"
    assert got["bad"] == "{{steps[0].output.nope}}"
